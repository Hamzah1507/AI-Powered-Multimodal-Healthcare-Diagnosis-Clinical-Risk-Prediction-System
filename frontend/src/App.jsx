import { useState } from 'react'
import axios from 'axios'
import './App.css'
import Welcome from './Welcome'
import Auth from './Auth'

const API = 'http://127.0.0.1:8000'

export default function App() {
  const [screen, setScreen] = useState('welcome')
  const [authMode, setAuthMode] = useState('login')
  const [user, setUser] = useState(null)

  const [module, setModule] = useState('xray')
  const [image, setImage] = useState(null)
  const [preview, setPreview] = useState(null)

  const [patient, setPatient] = useState({ name: '', age: '', gender: 'Male', id: '' })
  const [vitals, setVitals] = useState({
    pregnancies: '', glucose: '', blood_pressure: '',
    skin_thickness: '', insulin: '', bmi: '', diabetes_pedigree: '', age: ''
  })

  const [xrayResult, setXrayResult] = useState(null)
  const [vitalsResult, setVitalsResult] = useState(null)
  const [brainResult, setBrainResult] = useState(null)
  const [xrayHeatmap, setXrayHeatmap] = useState(null)
  const [brainHeatmap, setBrainHeatmap] = useState(null)

  const [loading, setLoading] = useState(false)
  const [gradcamLoading, setGradcamLoading] = useState(false)
  const [reportLoading, setReportLoading] = useState(false)
  const [error, setError] = useState(null)
  const [savedMsg, setSavedMsg] = useState(null)

  // ── Screen routing ───────────────────────────────────────────────
  if (screen === 'welcome') return (
    <Welcome
      onLogin={() => { setAuthMode('login'); setScreen('auth') }}
      onRegister={() => { setAuthMode('register'); setScreen('auth') }}
    />
  )
  if (screen === 'auth') return (
    <Auth
      mode={authMode}
      onSuccess={(u) => { setUser(u); setScreen('dashboard') }}
      onBack={() => setScreen('welcome')}
    />
  )

  // ── Helpers ──────────────────────────────────────────────────────
  const reset = () => {
    setImage(null); setPreview(null); setError(null); setSavedMsg(null)
    setXrayResult(null); setVitalsResult(null); setBrainResult(null)
    setXrayHeatmap(null); setBrainHeatmap(null)
    setVitals({ pregnancies: '', glucose: '', blood_pressure: '', skin_thickness: '', insulin: '', bmi: '', diabetes_pedigree: '', age: '' })
    setPatient({ name: '', age: '', gender: 'Male', id: '' })
  }

  const switchModule = (m) => { setModule(m); reset() }

  const handleImage = (e) => {
    const file = e.target.files[0]
    if (!file) return
    setImage(file); setPreview(URL.createObjectURL(file)); setError(null)
  }

  const savePrediction = async (data) => {
    try {
      await axios.post(`${API}/save-prediction`, data)
      setSavedMsg('✅ Patient record saved to database!')
      setTimeout(() => setSavedMsg(null), 4000)
    } catch { console.error('Failed to save prediction') }
  }

  const analyze = async () => {
    setError(null); setSavedMsg(null)
    if (!image) { setError('Please upload an image before running the diagnosis.'); return }
    if (module === 'xray' && (!vitals.glucose || !vitals.bmi || !vitals.age)) {
      setError('Please fill in required vitals: Glucose, BMI, and Age.'); return
    }
    setLoading(true)
    setXrayResult(null); setVitalsResult(null); setBrainResult(null)
    setXrayHeatmap(null); setBrainHeatmap(null)
    try {
      const imgForm = new FormData()
      imgForm.append('image', image)
      if (module === 'xray') {
        const vForm = new FormData()
        Object.keys(vitals).forEach(k => vForm.append(k, vitals[k] || 0))
        const [xr, vr] = await Promise.all([
          axios.post(`${API}/predict-xray`, imgForm),
          axios.post(`${API}/predict-vitals`, vForm)
        ])
        setXrayResult(xr.data); setVitalsResult(vr.data)
        await savePrediction({
          patient_id: patient.id || 'N/A', patient_name: patient.name || 'Unknown',
          patient_age: patient.age || 'N/A', patient_gender: patient.gender || 'Male',
          module: 'xray', diagnosis: xr.data.diagnosis, risk_score: xr.data.risk_score,
          probabilities: xr.data.probabilities, vitals_diagnosis: vr.data.diagnosis,
          vitals_risk_score: vr.data.risk_score, vitals_probabilities: vr.data.probabilities,
          saved_by: user?.username || 'Unknown'
        })
      } else {
        const br = await axios.post(`${API}/predict-brain`, imgForm)
        if (br.data.status === 'error') setError(br.data.message)
        else {
          setBrainResult(br.data)
          await savePrediction({
            patient_id: patient.id || 'N/A', patient_name: patient.name || 'Unknown',
            patient_age: patient.age || 'N/A', patient_gender: patient.gender || 'Male',
            module: 'brain', diagnosis: br.data.diagnosis, risk_score: br.data.risk_score,
            probabilities: br.data.probabilities, saved_by: user?.username || 'Unknown'
          })
        }
      }
    } catch { setError('Cannot connect to backend. Please make sure the server is running.') }
    setLoading(false)
  }

  const generateHeatmaps = async () => {
    if (!image) return
    setGradcamLoading(true)
    try {
      const imgForm = new FormData()
      imgForm.append('image', image)
      if (module === 'xray') {
        const res = await axios.post(`${API}/gradcam-xray`, imgForm)
        setXrayHeatmap(res.data.heatmap)
      } else {
        const res = await axios.post(`${API}/gradcam-brain`, imgForm)
        setBrainHeatmap(res.data.heatmap)
      }
    } catch { setError('Failed to generate heatmap!') }
    setGradcamLoading(false)
  }

  const downloadReport = async () => {
    if (!image) return
    setReportLoading(true)
    try {
      const form = new FormData()
      form.append('image', image)
      form.append('module', module)
      form.append('patient_name', patient.name || '')
      form.append('patient_id', patient.id || '')
      form.append('patient_age', patient.age || '')
      form.append('patient_gender', patient.gender || 'Male')
      if (module === 'xray' && xrayResult && vitalsResult) {
        form.append('xray_diagnosis', xrayResult.diagnosis)
        form.append('xray_risk_score', xrayResult.risk_score)
        form.append('xray_prob_normal', xrayResult.probabilities['Normal'])
        form.append('xray_prob_pneumonia', xrayResult.probabilities['Pneumonia'])
        form.append('vitals_diagnosis', vitalsResult.diagnosis)
        form.append('vitals_risk_score', vitalsResult.risk_score)
        form.append('vitals_prob_no_diabetes', vitalsResult.probabilities['No Diabetes'])
        form.append('vitals_prob_diabetes', vitalsResult.probabilities['Diabetes'])
        form.append('heatmap', xrayHeatmap || '')
      }
      if (module === 'brain' && brainResult) {
        form.append('brain_diagnosis', brainResult.diagnosis)
        form.append('brain_risk_score', brainResult.risk_score)
        form.append('brain_prob_glioma', brainResult.probabilities['Glioma'])
        form.append('brain_prob_meningioma', brainResult.probabilities['Meningioma'])
        form.append('brain_prob_no_tumor', brainResult.probabilities['No Tumor'])
        form.append('brain_prob_pituitary', brainResult.probabilities['Pituitary'])
        form.append('heatmap', brainHeatmap || '')
      }
      const res = await axios.post(`${API}/generate-report`, form, { responseType: 'blob' })
      const url = window.URL.createObjectURL(new Blob([res.data]))
      const link = document.createElement('a')
      link.href = url
      link.setAttribute('download', `MediAI_Report_${patient.name || 'Patient'}.pdf`)
      document.body.appendChild(link); link.click(); link.remove()
    } catch { setError('Failed to generate PDF report!') }
    setReportLoading(false)
  }

  // ── Risk helpers ─────────────────────────────────────────────────
  const riskColor = s => s >= 70 ? '#dc2626' : s >= 40 ? '#d97706' : '#059669'
  const riskBg = s => s >= 70 ? '#fef2f2' : s >= 40 ? '#fffbeb' : '#ecfdf5'
  const riskLabel = s => s >= 70 ? 'High Risk' : s >= 40 ? 'Moderate Risk' : 'Low Risk'
  const riskIcon = s => s >= 70 ? '🔴' : s >= 40 ? '🟡' : '🟢'

  // ── Static data ──────────────────────────────────────────────────
  const HERO_STATS = {
    xray: [['X-Ray Accuracy', '98%'], ['Diabetes Acc.', '78%'], ['X-Ray Model', 'ResNet-50'], ['Explainability', 'Grad-CAM'], ['Reports', 'PDF Export']],
    brain: [['MRI Accuracy', '94.75%'], ['Tumour Classes', '4 Types'], ['MRI Model', 'EfficientNet-B3'], ['Explainability', 'Grad-CAM'], ['Reports', 'PDF Export']]
  }
  const VITALS_FIELDS = [
    ['pregnancies', 'Pregnancies', '0', ''],
    ['glucose', 'Glucose (mg/dL)', '120', '*'],
    ['blood_pressure', 'Blood Pressure', '80', ''],
    ['skin_thickness', 'Skin Thickness', '20', ''],
    ['insulin', 'Insulin (μU/mL)', '80', ''],
    ['bmi', 'BMI (kg/m²)', '25.0', '*'],
    ['diabetes_pedigree', 'Diabetes Pedigree', '0.5', ''],
    ['age', 'Age (Years)', '30', '*'],
  ]
  const BRAIN_CONDITIONS = [
    ['#dc2626', '#fef2f2', '#dc262620', 'Glioma', 'Most common malignant brain tumour; arises from glial cells.'],
    ['#ea580c', '#fff7ed', '#ea580c20', 'Meningioma', 'Usually benign; originates from the meninges.'],
    ['#7c3aed', '#faf5ff', '#7c3aed20', 'Pituitary', 'Affects the pituitary gland; often benign.'],
    ['#059669', '#ecfdf5', '#05966920', 'No Tumor', 'No abnormality detected in the scan.'],
  ]

  // ── Sub-components ───────────────────────────────────────────────
  const ResultCard = ({ icon, title, result, color }) => (
    <div className="result-card" style={{ borderTopColor: color }}>
      <div className="result-card-header">
        <span className="result-card-icon">{icon}</span>
        <h3 className="result-card-title">{title}</h3>
      </div>
      <div className="dx-box" style={{ background: riskBg(result.risk_score), borderColor: riskColor(result.risk_score) + '30' }}>
        <p className="dx-label">Primary Diagnosis</p>
        <p className="dx-diagnosis">{result.diagnosis}</p>
        <div className="dx-risk-row">
          <span>{riskIcon(result.risk_score)}</span>
          <span className="dx-risk-label" style={{ color: riskColor(result.risk_score) }}>{riskLabel(result.risk_score)}</span>
          <span className="dx-risk-score">— Score: {result.risk_score}/100</span>
        </div>
      </div>
      <div className="risk-bar-track">
        <div className="risk-bar-fill" style={{ width: `${result.risk_score}%`, background: `linear-gradient(90deg,${color},${riskColor(result.risk_score)})` }} />
      </div>
      <p className="prob-section-label">Probability Breakdown</p>
      {Object.entries(result.probabilities).map(([d, p]) => (
        <div key={d} className="prob-row">
          <div className="prob-row-header">
            <span className="prob-class">{d}</span>
            <span className="prob-pct">{p}%</span>
          </div>
          <div className="prob-track">
            <div className="prob-fill" style={{ width: `${p}%`, background: color }} />
          </div>
        </div>
      ))}
    </div>
  )

  const PatientBadge = () => patient.name ? (
    <div className="patient-badge">
      <p className="patient-badge-label">Patient</p>
      <p className="patient-badge-name">{patient.name}</p>
      <p className="patient-badge-meta">
        {patient.id && `ID: ${patient.id} · `}
        {patient.age && `Age: ${patient.age} · `}
        {patient.gender}
      </p>
    </div>
  ) : null

  const HeatmapSection = ({ heatmap, color, borderColor }) => (
    <div className="heatmap-card" style={{ borderColor }}>
      <div className="heatmap-card-header">
        <div>
          <p className="heatmap-title">🔬 AI Attention Heatmap — Grad-CAM</p>
          <p className="heatmap-sub">Visualise which image regions influenced the AI diagnosis</p>
        </div>
        <button
          onClick={generateHeatmaps} disabled={gradcamLoading}
          className={`action-btn ${gradcamLoading ? '' : 'action-btn-heat'}`}
          style={gradcamLoading ? {} : { borderColor: color, color }}
        >
          {gradcamLoading ? '⏳ Generating…' : '🔥 Generate Heatmap'}
        </button>
      </div>

      {!heatmap && !gradcamLoading && (
        <div className="heatmap-empty">
          <span className="heatmap-empty-icon">🔬</span>
          <p className="heatmap-empty-text">Click "Generate Heatmap" to visualise AI attention</p>
          <p className="heatmap-empty-sub">Highlights the regions the model focused on</p>
        </div>
      )}
      {gradcamLoading && (
        <div className="heatmap-empty">
          <span className="heatmap-empty-icon">⏳</span>
          <p className="heatmap-empty-text">Generating heatmap, please wait…</p>
        </div>
      )}
      {heatmap && (
        <>
          <div className="heatmap-grid">
            <div className="heatmap-img-block">
              <p className="heatmap-img-label">Original Image</p>
              <img src={preview} alt="original" className="heatmap-img" />
            </div>
            <div className="heatmap-img-block">
              <p className="heatmap-img-label">AI Attention Heatmap</p>
              <img src={`data:image/jpeg;base64,${heatmap}`} alt="heatmap" className="heatmap-img" style={{ background: 'transparent' }} />
            </div>
          </div>
          <div className="heatmap-legend">
            <strong>📖 How to read:</strong>&nbsp; 🔴 <strong>Red/Yellow</strong> — areas of highest AI attention (likely abnormal). &nbsp;🔵 <strong>Blue</strong> — low relevance to the diagnosis.
          </div>
        </>
      )}
    </div>
  )

  // ── Render ───────────────────────────────────────────────────────
  return (
    <div className="app-shell">

      {/* ── Navbar ──────────────────────────────────────────────── */}
      <nav className="navbar">
        {/* Brand */}
        <div className="navbar-brand">
          <div className="navbar-logo">🏥</div>
          <div>
            <div className="navbar-title">MediAI Diagnostics</div>
            <div className="navbar-sub">AI-Powered Clinical Decision Support</div>
          </div>
        </div>

        {/* Centre tabs */}
        <div className="navbar-center">
          {[['xray', '🫁', 'Chest X-Ray'], ['brain', '🧠', 'Brain MRI']].map(([m, icon, label]) => (
            <button
              key={m} onClick={() => switchModule(m)}
              className={`tab-btn ${module === m ? (m === 'brain' ? 'tab-active-purple' : 'tab-active-blue') : ''}`}
            >
              {icon} {label}
            </button>
          ))}
        </div>

        {/* Right actions */}
        <div className="navbar-right">
          {user && (
            <div className="user-pill">
              <span className="user-pill-dot" />
              <span style={{ fontSize: '13px', fontWeight: '700', color: '#065f46' }}>{user.username}</span>
            </div>
          )}
          <div className="nav-sep" />
          <button className="logout-btn" onClick={() => { reset(); setScreen('welcome'); setUser(null) }}>
            🚪 Logout
          </button>
        </div>
      </nav>

      {/* ── Hero Banner ─────────────────────────────────────────── */}
      <div className={`hero ${module}`}>
        <div className="hero-inner">
          <div className="hero-eyebrow">
            <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: 'rgba(255,255,255,0.8)', display: 'inline-block' }} />
            GLS University Capstone 2025–26
          </div>
          <h2 className="hero-title">
            {module === 'xray' ? '🫁 Chest X-Ray + Diabetes Analysis' : '🧠 Brain MRI Tumour Detection'}
          </h2>
          <p className="hero-sub">
            {module === 'xray'
              ? 'Upload a chest X-ray and enter patient vitals for AI-powered pneumonia detection and diabetes risk assessment.'
              : 'Upload a brain MRI scan for AI-powered tumour classification and comprehensive risk assessment.'}
          </p>
          <div className="hero-stats">
            {HERO_STATS[module].map(([label, value]) => (
              <div key={label} className="hero-stat">
                <p className="hero-stat-label">{label}</p>
                <p className="hero-stat-value">{value}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Page Body ───────────────────────────────────────────── */}
      <div className="page-body">

        {error && <div className="alert alert-error">⚠️ {error}</div>}
        {savedMsg && <div className="alert alert-success">{savedMsg}</div>}

        {/* ── Input Grid ────────────────────────────────────────── */}
        <div className="main-grid">

          {/* ── LEFT COLUMN ─────────────────────────────────────── */}
          <div>
            {/* Patient Info */}
            <div className="card">
              <div className="card-header">
                <div className="card-icon card-icon-blue">👤</div>
                <div>
                  <p className="card-title">Patient Information</p>
                  <p className="card-sub">Enter patient demographic details</p>
                </div>
              </div>
              <div className="form-grid-2">
                <div className="form-group span-2">
                  <label className="form-label">Full Name</label>
                  <input className="form-input" placeholder="Enter patient full name" value={patient.name}
                    onChange={e => setPatient({ ...patient, name: e.target.value })} />
                </div>
                <div className="form-group">
                  <label className="form-label">Patient ID</label>
                  <input className="form-input" placeholder="PT-001" value={patient.id}
                    onChange={e => setPatient({ ...patient, id: e.target.value })} />
                </div>
                <div className="form-group">
                  <label className="form-label">Age</label>
                  <input className="form-input" type="number" placeholder="Years" value={patient.age}
                    onChange={e => setPatient({ ...patient, age: e.target.value })} />
                </div>
                <div className="form-group span-2">
                  <label className="form-label">Gender</label>
                  <select className="form-input" value={patient.gender}
                    onChange={e => setPatient({ ...patient, gender: e.target.value })}>
                    <option>Male</option><option>Female</option><option>Other</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Image Upload */}
            <div className="card">
              <div className="card-header">
                <div className={`card-icon ${module === 'brain' ? 'card-icon-purple' : 'card-icon-blue'}`}>
                  {module === 'xray' ? '📷' : '🧠'}
                </div>
                <div>
                  <p className="card-title">{module === 'xray' ? 'Chest X-Ray Image' : 'Brain MRI Scan'}</p>
                  <p className="card-sub">Upload your medical imaging scan</p>
                </div>
              </div>
              <label className="upload-zone">
                <input type="file" accept="image/*" onChange={handleImage} style={{ display: 'none' }} />
                {preview
                  ? <img src={preview} alt="preview" className="upload-preview" />
                  : <>
                    <span className="upload-icon">🖼️</span>
                    <p className="upload-hint-main">Click to upload scan image</p>
                    <p className="upload-hint-sub">PNG, JPG, JPEG — Max 10 MB</p>
                  </>
                }
              </label>
              {module === 'brain' && (
                <div className="module-warning">⚠️ Only brain MRI scans are accepted for this module.</div>
              )}
            </div>
          </div>

          {/* ── RIGHT COLUMN ────────────────────────────────────── */}
          <div>
            {/* Vitals (X-Ray mode) */}
            {module === 'xray' && (
              <div className="card">
                <div className="card-header">
                  <div className="card-icon card-icon-green">💉</div>
                  <div>
                    <p className="card-title">Patient Vitals</p>
                    <p className="card-sub">Lab values for diabetes risk assessment</p>
                  </div>
                </div>
                <div className="form-grid-2">
                  {VITALS_FIELDS.map(([key, label, ph, req]) => (
                    <div key={key} className="form-group">
                      <label className="form-label">
                        {label}{req && <span className="req">*</span>}
                      </label>
                      <input type="number" className="form-input" placeholder={ph}
                        value={vitals[key]} onChange={e => setVitals({ ...vitals, [key]: e.target.value })} />
                    </div>
                  ))}
                </div>
                <p style={{ fontSize: '11px', color: 'var(--text-4)', marginTop: '6px' }}>
                  <span style={{ color: 'var(--red)' }}>*</span> Required fields
                </p>
              </div>
            )}

            {/* Detectable conditions (Brain mode) */}
            {module === 'brain' && (
              <div className="card">
                <div className="card-header">
                  <div className="card-icon card-icon-purple">🔍</div>
                  <div>
                    <p className="card-title">Detectable Conditions</p>
                    <p className="card-sub">4 tumour types supported by this model</p>
                  </div>
                </div>
                {BRAIN_CONDITIONS.map(([dot, bg, border, name, desc]) => (
                  <div key={name} className="condition-item" style={{ background: bg, borderColor: border }}>
                    <span className="condition-dot" style={{ background: dot }} />
                    <div>
                      <p className="condition-name">{name}</p>
                      <p className="condition-desc">{desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Analyze button */}
            <button
              onClick={analyze} disabled={loading}
              className={`analyze-btn ${module === 'brain' ? 'purple' : 'blue'}`}
            >
              {loading
                ? <><span>⏳</span> Analysing Patient Data…</>
                : <><span>🔍</span> Run AI Diagnosis</>
              }
            </button>
          </div>
        </div>

        {/* ── X-Ray Results ─────────────────────────────────────── */}
        {xrayResult && vitalsResult && (
          <section className="results-section">
            <div className="results-header">
              <div>
                <h2 className="results-title">📊 Diagnosis Results</h2>
                <p className="results-date">
                  {new Date().toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}
                </p>
              </div>
              <PatientBadge />
            </div>

            <div className="result-row">
              <ResultCard icon="🫁" title="Chest X-Ray Analysis" result={xrayResult} color="#2563eb" />
              <ResultCard icon="🩸" title="Diabetes Risk Analysis" result={vitalsResult} color="#059669" />
            </div>

            <div className="action-row">
              <button onClick={downloadReport} disabled={reportLoading}
                className={`action-btn ${reportLoading ? '' : 'action-btn-pdf'}`}>
                {reportLoading ? '⏳ Generating PDF…' : '📄 Download PDF Report'}
              </button>
            </div>

            <div className="disclaimer">
              <span>⚠️</span>
              <p><strong>Medical Disclaimer:</strong> This AI-assisted result is for clinical decision support only. Always consult a qualified medical professional before making any clinical decisions.</p>
            </div>

            <HeatmapSection heatmap={xrayHeatmap} color="#2563eb" borderColor="#bfdbfe" />
          </section>
        )}

        {/* ── Brain Results ──────────────────────────────────────── */}
        {brainResult && (
          <section className="results-section">
            <div className="results-header">
              <div>
                <h2 className="results-title">📊 Brain MRI Results</h2>
                <p className="results-date">
                  {new Date().toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}
                </p>
              </div>
              <PatientBadge />
            </div>

            <ResultCard icon="🧠" title="Brain Tumour Analysis" result={brainResult} color="#7c3aed" />

            <div className="action-row">
              <button onClick={downloadReport} disabled={reportLoading}
                className={`action-btn ${reportLoading ? '' : 'action-btn-pdf'}`}
                style={reportLoading ? {} : { background: 'linear-gradient(135deg,#5b21b6,#7c3aed)', boxShadow: '0 4px 14px rgba(124,58,237,0.35)' }}>
                {reportLoading ? '⏳ Generating PDF…' : '📄 Download PDF Report'}
              </button>
            </div>

            <div className="disclaimer">
              <span>⚠️</span>
              <p><strong>Medical Disclaimer:</strong> This AI-assisted result is for clinical decision support only. Always consult a qualified medical professional before making any clinical decisions.</p>
            </div>

            <HeatmapSection heatmap={brainHeatmap} color="#7c3aed" borderColor="#ddd6fe" />
          </section>
        )}

        {/* Footer */}
        <div className="page-footer">
          <p>MediAI Diagnostics · GLS University Capstone Project 2025–26 · Integrated MSc(IT) Programme</p>
        </div>
      </div>
    </div>
  )
}