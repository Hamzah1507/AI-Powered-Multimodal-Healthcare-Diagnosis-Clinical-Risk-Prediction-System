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
  const [vitals, setVitals] = useState({ pregnancies: '', glucose: '', blood_pressure: '', skin_thickness: '', insulin: '', bmi: '', diabetes_pedigree: '', age: '' })
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

  if (screen === 'welcome') return <Welcome onLogin={() => { setAuthMode('login'); setScreen('auth') }} onRegister={() => { setAuthMode('register'); setScreen('auth') }} />
  if (screen === 'auth') return <Auth mode={authMode} onSuccess={(u) => { setUser(u); setScreen('dashboard') }} onBack={() => setScreen('welcome')} />

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
      setSavedMsg('Record saved to database')
      setTimeout(() => setSavedMsg(null), 4000)
    } catch { console.error('Failed to save') }
  }

  const analyze = async () => {
    setError(null); setSavedMsg(null)
    if (!image) { setError('Please upload an image first'); return }
    if (module === 'xray' && (!vitals.glucose || !vitals.bmi || !vitals.age)) { setError('Please fill Glucose, BMI and Age fields'); return }
    setLoading(true)
    setXrayResult(null); setVitalsResult(null); setBrainResult(null); setXrayHeatmap(null); setBrainHeatmap(null)
    try {
      const imgForm = new FormData(); imgForm.append('image', image)
      if (module === 'xray') {
        const vForm = new FormData()
        Object.keys(vitals).forEach(k => vForm.append(k, vitals[k] || 0))
        const [xr, vr] = await Promise.all([axios.post(`${API}/predict-xray`, imgForm), axios.post(`${API}/predict-vitals`, vForm)])
        setXrayResult(xr.data); setVitalsResult(vr.data)
        await savePrediction({ patient_id: patient.id || 'N/A', patient_name: patient.name || 'Unknown', patient_age: patient.age || 'N/A', patient_gender: patient.gender, module: 'xray', diagnosis: xr.data.diagnosis, risk_score: xr.data.risk_score, probabilities: xr.data.probabilities, vitals_diagnosis: vr.data.diagnosis, vitals_risk_score: vr.data.risk_score, vitals_probabilities: vr.data.probabilities, saved_by: user?.username || 'Unknown' })
      } else {
        const br = await axios.post(`${API}/predict-brain`, imgForm)
        if (br.data.status === 'error') setError(br.data.message)
        else { setBrainResult(br.data); await savePrediction({ patient_id: patient.id || 'N/A', patient_name: patient.name || 'Unknown', patient_age: patient.age || 'N/A', patient_gender: patient.gender, module: 'brain', diagnosis: br.data.diagnosis, risk_score: br.data.risk_score, probabilities: br.data.probabilities, saved_by: user?.username || 'Unknown' }) }
      }
    } catch { setError('Cannot connect to backend. Make sure the server is running.') }
    setLoading(false)
  }

  const generateHeatmaps = async () => {
    if (!image) return; setGradcamLoading(true)
    try {
      const imgForm = new FormData(); imgForm.append('image', image)
      if (module === 'xray') { const res = await axios.post(`${API}/gradcam-xray`, imgForm); setXrayHeatmap(res.data.heatmap) }
      else { const res = await axios.post(`${API}/gradcam-brain`, imgForm); setBrainHeatmap(res.data.heatmap) }
    } catch { setError('Failed to generate heatmap') }
    setGradcamLoading(false)
  }

  const downloadReport = async () => {
    if (!image) return; setReportLoading(true)
    try {
      const form = new FormData()
      form.append('image', image); form.append('module', module)
      form.append('patient_name', patient.name || ''); form.append('patient_id', patient.id || '')
      form.append('patient_age', patient.age || ''); form.append('patient_gender', patient.gender || 'Male')
      if (module === 'xray' && xrayResult && vitalsResult) {
        form.append('xray_diagnosis', xrayResult.diagnosis); form.append('xray_risk_score', xrayResult.risk_score)
        form.append('xray_prob_normal', xrayResult.probabilities['Normal']); form.append('xray_prob_pneumonia', xrayResult.probabilities['Pneumonia'])
        form.append('vitals_diagnosis', vitalsResult.diagnosis); form.append('vitals_risk_score', vitalsResult.risk_score)
        form.append('vitals_prob_no_diabetes', vitalsResult.probabilities['No Diabetes']); form.append('vitals_prob_diabetes', vitalsResult.probabilities['Diabetes'])
        form.append('heatmap', xrayHeatmap || '')
      }
      if (module === 'brain' && brainResult) {
        form.append('brain_diagnosis', brainResult.diagnosis); form.append('brain_risk_score', brainResult.risk_score)
        form.append('brain_prob_glioma', brainResult.probabilities['Glioma']); form.append('brain_prob_meningioma', brainResult.probabilities['Meningioma'])
        form.append('brain_prob_no_tumor', brainResult.probabilities['No Tumor']); form.append('brain_prob_pituitary', brainResult.probabilities['Pituitary'])
        form.append('heatmap', brainHeatmap || '')
      }
      const res = await axios.post(`${API}/generate-report`, form, { responseType: 'blob' })
      const url = window.URL.createObjectURL(new Blob([res.data]))
      const link = document.createElement('a'); link.href = url
      link.setAttribute('download', `MediAI_${patient.name || 'Patient'}_Report.pdf`)
      document.body.appendChild(link); link.click(); link.remove()
    } catch { setError('Failed to generate PDF report') }
    setReportLoading(false)
  }

  const isXray = module === 'xray'
  const accent = isXray ? '#2563eb' : '#7c3aed'
  const accentGrad = isXray ? 'linear-gradient(135deg, #1e40af, #2563eb)' : 'linear-gradient(135deg, #5b21b6, #7c3aed)'
  const riskColor = (s) => s >= 70 ? '#ef4444' : s >= 40 ? '#f59e0b' : '#10b981'
  const riskBg = (s) => s >= 70 ? '#fef2f2' : s >= 40 ? '#fffbeb' : '#f0fdf4'
  const riskBorder = (s) => s >= 70 ? '#fecaca' : s >= 40 ? '#fde68a' : '#a7f3d0'
  const riskLabel = (s) => s >= 70 ? 'High Risk' : s >= 40 ? 'Moderate' : 'Low Risk'

  const inp = { width: '100%', padding: '10px 12px', borderRadius: '8px', border: '1.5px solid #e8ecf4', background: '#fafbfd', color: '#0f172a', fontSize: '13px', boxSizing: 'border-box', outline: 'none', fontFamily: 'inherit', transition: 'border-color 0.2s, box-shadow 0.2s' }
  const lbl = { fontSize: '10px', fontWeight: '700', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '1px', display: 'block', marginBottom: '5px' }
  const card = { background: '#fff', borderRadius: '16px', padding: '22px', marginBottom: '16px', border: '1px solid #f0f2f5', boxShadow: '0 1px 3px rgba(0,0,0,0.04), 0 4px 16px rgba(0,0,0,0.03)' }

  const ResultCard = ({ icon, title, result, color }) => (
    <div style={{ background: '#fff', borderRadius: '16px', padding: '24px', flex: 1, border: '1px solid #f0f2f5', boxShadow: '0 2px 8px rgba(0,0,0,0.05)', position: 'relative', overflow: 'hidden' }}>
      <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '3px', background: `linear-gradient(90deg, ${color}, ${color}88)` }} />
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '18px' }}>
        <div style={{ width: '36px', height: '36px', borderRadius: '10px', background: `${color}15`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '18px' }}>{icon}</div>
        <h3 style={{ color: '#0f172a', fontSize: '14px', fontWeight: '700', letterSpacing: '-0.2px' }}>{title}</h3>
      </div>
      <div style={{ background: riskBg(result.risk_score), borderRadius: '12px', padding: '16px', marginBottom: '18px', border: `1px solid ${riskBorder(result.risk_score)}` }}>
        <p style={{ color: '#94a3b8', fontSize: '10px', fontWeight: '700', letterSpacing: '1.5px', marginBottom: '6px' }}>DIAGNOSIS</p>
        <p style={{ color: '#0f172a', fontSize: '22px', fontWeight: '800', letterSpacing: '-0.5px', lineHeight: 1.1 }}>{result.diagnosis}</p>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '10px' }}>
          <span style={{ background: riskColor(result.risk_score), color: '#fff', fontSize: '10px', fontWeight: '800', padding: '3px 10px', borderRadius: '100px', letterSpacing: '0.5px' }}>{riskLabel(result.risk_score)}</span>
          <span style={{ color: '#94a3b8', fontSize: '12px' }}>Score: <strong style={{ color: riskColor(result.risk_score) }}>{result.risk_score}</strong>/100</span>
        </div>
      </div>
      <div style={{ background: '#f8fafc', borderRadius: '8px', height: '5px', marginBottom: '18px', overflow: 'hidden' }}>
        <div style={{ width: `${result.risk_score}%`, height: '100%', background: `linear-gradient(90deg, ${color}, ${riskColor(result.risk_score)})`, borderRadius: '8px', transition: 'width 1.4s cubic-bezier(0.4,0,0.2,1)' }} />
      </div>
      <p style={{ color: '#94a3b8', fontSize: '10px', fontWeight: '700', letterSpacing: '1.5px', marginBottom: '12px' }}>PROBABILITY BREAKDOWN</p>
      {Object.entries(result.probabilities).map(([d, p]) => (
        <div key={d} style={{ marginBottom: '10px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
            <span style={{ color: '#475569', fontSize: '12px', fontWeight: '500' }}>{d}</span>
            <span style={{ color: '#0f172a', fontWeight: '800', fontSize: '12px' }}>{p}%</span>
          </div>
          <div style={{ background: '#f1f5f9', borderRadius: '6px', height: '5px', overflow: 'hidden' }}>
            <div style={{ width: `${p}%`, height: '100%', background: color, borderRadius: '6px', transition: 'width 1.4s cubic-bezier(0.4,0,0.2,1)' }} />
          </div>
        </div>
      ))}
    </div>
  )

  const HeatmapSection = ({ heatmap, color }) => (
    <div style={{ ...card, marginTop: '20px', border: `1px solid ${color}22` }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
        <div>
          <h3 style={{ fontSize: '14px', fontWeight: '700', color: '#0f172a', letterSpacing: '-0.2px' }}>Grad-CAM Heatmap</h3>
          <p style={{ color: '#94a3b8', fontSize: '12px', marginTop: '2px' }}>AI attention visualization</p>
        </div>
        <button onClick={generateHeatmaps} disabled={gradcamLoading}
          onMouseEnter={e => !gradcamLoading && (e.currentTarget.style.opacity = '0.85')}
          onMouseLeave={e => (e.currentTarget.style.opacity = '1')}
          style={{ padding: '8px 18px', border: 'none', borderRadius: '8px', background: gradcamLoading ? '#f1f5f9' : color, color: gradcamLoading ? '#94a3b8' : 'white', fontWeight: '600', fontSize: '12px', cursor: gradcamLoading ? 'not-allowed' : 'pointer', transition: 'all 0.2s' }}>
          {gradcamLoading ? 'Generating...' : 'Generate Heatmap'}
        </button>
      </div>
      {!heatmap && !gradcamLoading && (
        <div style={{ background: '#f8fafc', borderRadius: '12px', padding: '32px', textAlign: 'center', border: '2px dashed #e8ecf4' }}>
          <p style={{ color: '#64748b', fontSize: '13px', fontWeight: '600' }}>Generate heatmap to visualize AI attention regions</p>
          <p style={{ color: '#94a3b8', fontSize: '11px', marginTop: '4px' }}>Red/yellow = high attention · Blue = low attention</p>
        </div>
      )}
      {gradcamLoading && (
        <div style={{ background: '#f8fafc', borderRadius: '12px', padding: '32px', textAlign: 'center' }}>
          <p style={{ color: '#64748b', fontSize: '13px', fontWeight: '600' }}>Computing Grad-CAM heatmap...</p>
        </div>
      )}
      {heatmap && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '12px' }}>
            {[['Original Scan', preview, false], ['AI Attention Map', `data:image/jpeg;base64,${heatmap}`, true]].map(([label, src, isHeat]) => (
              <div key={label} style={{ textAlign: 'center' }}>
                <p style={{ color: '#94a3b8', fontSize: '10px', fontWeight: '700', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '1px' }}>{label}</p>
                <img src={src} alt={label} style={{ width: '100%', borderRadius: '10px', objectFit: 'contain', maxHeight: '220px', background: '#000' }} />
              </div>
            ))}
          </div>
          <div style={{ background: '#f8fafc', borderRadius: '8px', padding: '10px 14px', border: '1px solid #f0f2f5' }}>
            <p style={{ color: '#64748b', fontSize: '11px' }}>Red/yellow regions indicate where the AI detected anomalies · Blue regions have lower diagnostic significance</p>
          </div>
        </>
      )}
    </div>
  )

  return (
    <div style={{ minHeight: '100vh', background: '#f4f6f9', fontFamily: "'Helvetica Neue', Arial, sans-serif" }}>

      {/* Navbar */}
      <nav style={{ background: '#fff', borderBottom: '1px solid #edf0f4', padding: '0 24px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', height: '54px', position: 'sticky', top: 0, zIndex: 100, boxShadow: '0 1px 0 #edf0f4' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{ width: '28px', height: '28px', borderRadius: '7px', background: 'linear-gradient(135deg, #2563eb, #0ea5e9)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white', fontWeight: '900', fontSize: '13px' }}>+</div>
          <div>
            <p style={{ fontSize: '13px', fontWeight: '800', color: '#0f172a', letterSpacing: '-0.3px', lineHeight: 1 }}>MediAI Diagnostics</p>
            <p style={{ fontSize: '9px', color: '#94a3b8', letterSpacing: '0.5px', textTransform: 'uppercase' }}>Clinical Decision Support</p>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          {[['xray', 'Chest X-Ray'], ['brain', 'Brain MRI']].map(([m, label]) => (
            <button key={m} onClick={() => switchModule(m)}
              style={{ padding: '6px 14px', borderRadius: '7px', border: module === m ? 'none' : '1.5px solid #edf0f4', cursor: 'pointer', fontWeight: '600', fontSize: '12px', background: module === m ? (m === 'brain' ? '#7c3aed' : '#2563eb') : '#fafbfd', color: module === m ? 'white' : '#64748b', transition: 'all 0.2s' }}>
              {label}
            </button>
          ))}

          <div style={{ width: '1px', height: '20px', background: '#edf0f4', margin: '0 4px' }} />

          {user && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '7px', padding: '5px 10px', background: '#f0fdf4', borderRadius: '7px', border: '1px solid #bbf7d0' }}>
              <div style={{ width: '20px', height: '20px', borderRadius: '50%', background: '#10b981', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white', fontSize: '10px', fontWeight: '800' }}>{user.username?.[0]?.toUpperCase()}</div>
              <span style={{ color: '#059669', fontWeight: '700', fontSize: '12px' }}>{user.username}</span>
            </div>
          )}

          <button onClick={() => { reset(); setScreen('welcome'); setUser(null) }}
            onMouseEnter={e => e.currentTarget.style.background = '#fef2f2'}
            onMouseLeave={e => e.currentTarget.style.background = '#fff'}
            style={{ padding: '6px 12px', borderRadius: '7px', border: '1.5px solid #fecaca', cursor: 'pointer', fontWeight: '600', fontSize: '12px', background: '#fff', color: '#dc2626', transition: 'all 0.2s' }}>
            Sign Out
          </button>
        </div>
      </nav>

      {/* Module Banner */}
      <div style={{ background: accentGrad, padding: '20px 24px' }}>
        <div style={{ maxWidth: '1100px', margin: '0 auto', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '16px', flexWrap: 'wrap' }}>
          <div>
            <h2 style={{ fontSize: '18px', fontWeight: '800', color: 'white', letterSpacing: '-0.4px', marginBottom: '3px' }}>
              {isXray ? 'Chest X-Ray + Diabetes Analysis' : 'Brain MRI Tumor Detection'}
            </h2>
            <p style={{ color: 'rgba(255,255,255,0.6)', fontSize: '12px' }}>
              {isXray ? 'ResNet-50 pneumonia detection + Diabetes risk prediction' : 'EfficientNet-B3 powered 4-class brain tumor classification'}
            </p>
          </div>
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            {(isXray ? [['98%', 'X-Ray Acc.'], ['78%', 'Diabetes Acc.'], ['ResNet-50', 'Model'], ['Grad-CAM', 'XAI'], ['PDF', 'Reports']] : [['94.75%', 'MRI Acc.'], ['4', 'Tumor Types'], ['EfficientNet-B3', 'Model'], ['Grad-CAM', 'XAI'], ['PDF', 'Reports']]).map(([v, l]) => (
              <div key={l} style={{ background: 'rgba(255,255,255,0.12)', border: '1px solid rgba(255,255,255,0.18)', borderRadius: '8px', padding: '7px 12px', backdropFilter: 'blur(4px)' }}>
                <p style={{ color: 'white', fontWeight: '800', fontSize: '13px', lineHeight: 1 }}>{v}</p>
                <p style={{ color: 'rgba(255,255,255,0.55)', fontSize: '9px', letterSpacing: '0.8px', textTransform: 'uppercase', marginTop: '2px' }}>{l}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Main */}
      <div style={{ maxWidth: '1100px', margin: '0 auto', padding: '24px' }}>

        {/* Alerts */}
        {error && (
          <div style={{ background: '#fef2f2', border: '1px solid #fecaca', padding: '11px 16px', borderRadius: '10px', marginBottom: '18px', color: '#dc2626', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontWeight: '700', fontSize: '14px' }}>⚠</span> {error}
            <button onClick={() => setError(null)} style={{ marginLeft: 'auto', background: 'none', border: 'none', cursor: 'pointer', color: '#dc2626', fontSize: '16px', padding: 0 }}>×</button>
          </div>
        )}
        {savedMsg && (
          <div style={{ background: '#f0fdf4', border: '1px solid #bbf7d0', padding: '11px 16px', borderRadius: '10px', marginBottom: '18px', color: '#059669', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span>✓</span> {savedMsg}
          </div>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '18px' }}>

          {/* ── LEFT ── */}
          <div>
            {/* Patient Card */}
            <div style={card}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px', paddingBottom: '13px', borderBottom: '1px solid #f4f6f9' }}>
                <div style={{ width: '32px', height: '32px', borderRadius: '8px', background: '#eff6ff', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '14px' }}>👤</div>
                <div>
                  <p style={{ fontSize: '13px', fontWeight: '700', color: '#0f172a', letterSpacing: '-0.2px' }}>Patient Information</p>
                  <p style={{ fontSize: '11px', color: '#94a3b8' }}>Enter patient details</p>
                </div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                <div style={{ gridColumn: '1/-1' }}>
                  <label style={lbl}>Full Name</label>
                  <input style={inp} value={patient.name} onChange={e => setPatient({ ...patient, name: e.target.value })}
                    onFocus={e => { e.target.style.borderColor = accent; e.target.style.boxShadow = `0 0 0 3px ${accent}15` }}
                    onBlur={e => { e.target.style.borderColor = '#e8ecf4'; e.target.style.boxShadow = 'none' }} />
                </div>
                <div>
                  <label style={lbl}>Patient ID</label>
                  <input style={inp} placeholder="PT-001" value={patient.id} onChange={e => setPatient({ ...patient, id: e.target.value })}
                    onFocus={e => { e.target.style.borderColor = accent; e.target.style.boxShadow = `0 0 0 3px ${accent}15` }}
                    onBlur={e => { e.target.style.borderColor = '#e8ecf4'; e.target.style.boxShadow = 'none' }} />
                </div>
                <div>
                  <label style={lbl}>Age</label>
                  <input style={inp} type="number" value={patient.age} onChange={e => setPatient({ ...patient, age: e.target.value })}
                    onFocus={e => { e.target.style.borderColor = accent; e.target.style.boxShadow = `0 0 0 3px ${accent}15` }}
                    onBlur={e => { e.target.style.borderColor = '#e8ecf4'; e.target.style.boxShadow = 'none' }} />
                </div>
                <div>
                  <label style={lbl}>Gender</label>
                  <select style={inp} value={patient.gender} onChange={e => setPatient({ ...patient, gender: e.target.value })}>
                    <option>Male</option><option>Female</option><option>Other</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Upload Card */}
            <div style={card}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px', paddingBottom: '13px', borderBottom: '1px solid #f4f6f9' }}>
                <div style={{ width: '32px', height: '32px', borderRadius: '8px', background: isXray ? '#eff6ff' : '#f5f3ff', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '14px' }}>
                  {isXray ? '🫁' : '🧠'}
                </div>
                <div>
                  <p style={{ fontSize: '13px', fontWeight: '700', color: '#0f172a' }}>{isXray ? 'Chest X-Ray Image' : 'Brain MRI Scan'}</p>
                  <p style={{ fontSize: '11px', color: '#94a3b8' }}>Upload medical scan image</p>
                </div>
              </div>
              <label style={{ display: 'block', border: `2px dashed ${preview ? accent : '#e8ecf4'}`, borderRadius: '12px', padding: '20px', textAlign: 'center', cursor: 'pointer', background: preview ? `${accent}06` : '#fafbfd', transition: 'all 0.2s' }}>
                <input type="file" accept="image/*" onChange={handleImage} style={{ display: 'none' }} />
                {preview ? (
                  <img src={preview} alt="preview" style={{ maxWidth: '100%', maxHeight: '160px', borderRadius: '8px', objectFit: 'contain' }} />
                ) : (
                  <>
                    <div style={{ width: '36px', height: '36px', borderRadius: '10px', background: '#f0f2f5', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '16px', margin: '0 auto 8px' }}>📁</div>
                    <p style={{ color: '#475569', fontSize: '12px', fontWeight: '600' }}>Click to upload image</p>
                    <p style={{ color: '#94a3b8', fontSize: '11px', marginTop: '2px' }}>PNG, JPG, JPEG · Max 10MB</p>
                  </>
                )}
              </label>
              {!isXray && (
                <div style={{ background: '#faf5ff', border: '1px solid #e9d5ff', borderRadius: '8px', padding: '8px 12px', marginTop: '10px' }}>
                  <p style={{ color: '#7c3aed', fontSize: '11px', fontWeight: '600' }}>Only Brain MRI scans accepted</p>
                </div>
              )}
            </div>
          </div>

          {/* ── RIGHT ── */}
          <div>
            {isXray ? (
              <div style={card}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px', paddingBottom: '13px', borderBottom: '1px solid #f4f6f9' }}>
                  <div style={{ width: '32px', height: '32px', borderRadius: '8px', background: '#f0fdf4', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '14px' }}>💉</div>
                  <div>
                    <p style={{ fontSize: '13px', fontWeight: '700', color: '#0f172a' }}>Patient Vitals</p>
                    <p style={{ fontSize: '11px', color: '#94a3b8' }}>Lab values for diabetes risk prediction</p>
                  </div>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                  {[['pregnancies', 'Pregnancies', ''], ['glucose', 'Glucose (mg/dL)', '*'], ['blood_pressure', 'Blood Pressure', ''], ['skin_thickness', 'Skin Thickness', ''], ['insulin', 'Insulin (μU/mL)', ''], ['bmi', 'BMI (kg/m²)', '*'], ['diabetes_pedigree', 'Pedigree Function', ''], ['age', 'Age (Years)', '*']].map(([key, label, req]) => (
                    <div key={key}>
                      <label style={lbl}>{label} {req && <span style={{ color: '#ef4444' }}>{req}</span>}</label>
                      <input type="number" style={inp} value={vitals[key]}
                        onChange={e => setVitals({ ...vitals, [key]: e.target.value })}
                        onFocus={e => { e.target.style.borderColor = '#10b981'; e.target.style.boxShadow = '0 0 0 3px #10b98115' }}
                        onBlur={e => { e.target.style.borderColor = '#e8ecf4'; e.target.style.boxShadow = 'none' }} />
                    </div>
                  ))}
                </div>
                <p style={{ color: '#94a3b8', fontSize: '10px', marginTop: '10px' }}>* Required fields</p>
              </div>
            ) : (
              <div style={card}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px', paddingBottom: '13px', borderBottom: '1px solid #f4f6f9' }}>
                  <div style={{ width: '32px', height: '32px', borderRadius: '8px', background: '#f5f3ff', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '14px' }}>🔬</div>
                  <div>
                    <p style={{ fontSize: '13px', fontWeight: '700', color: '#0f172a' }}>Detectable Conditions</p>
                    <p style={{ fontSize: '11px', color: '#94a3b8' }}>4 tumor classifications supported</p>
                  </div>
                </div>
                {[
                  ['Glioma', 'Most common malignant brain tumor. Originates in glial cells.', '#ef4444', '#fef2f2', '#fecaca'],
                  ['Meningioma', 'Usually benign. Arises from meninges surrounding the brain.', '#f59e0b', '#fffbeb', '#fde68a'],
                  ['Pituitary', 'Affects the pituitary gland and hormone regulation.', '#8b5cf6', '#f5f3ff', '#ddd6fe'],
                  ['No Tumor', 'No abnormality detected in the brain MRI scan.', '#10b981', '#f0fdf4', '#a7f3d0'],
                ].map(([name, desc, color, bg, border]) => (
                  <div key={name} style={{ display: 'flex', gap: '12px', padding: '11px 13px', borderRadius: '10px', marginBottom: '8px', background: bg, border: `1px solid ${border}` }}>
                    <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: color, flexShrink: 0, marginTop: '5px' }} />
                    <div>
                      <p style={{ color: '#0f172a', fontWeight: '700', fontSize: '12px' }}>{name}</p>
                      <p style={{ color: '#64748b', fontSize: '11px', marginTop: '2px', lineHeight: 1.5 }}>{desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Analyze Button */}
            <button onClick={analyze} disabled={loading}
              onMouseEnter={e => !loading && (e.currentTarget.style.opacity = '0.88')}
              onMouseLeave={e => (e.currentTarget.style.opacity = '1')}
              style={{ width: '100%', padding: '13px', border: 'none', borderRadius: '10px', fontSize: '13px', fontWeight: '700', cursor: loading ? 'not-allowed' : 'pointer', background: loading ? '#f0f2f5' : accentGrad, color: loading ? '#94a3b8' : 'white', boxShadow: loading ? 'none' : `0 4px 14px ${accent}35`, transition: 'all 0.2s', letterSpacing: '0.1px' }}>
              {loading ? 'Analyzing...' : 'Run AI Diagnosis'}
            </button>
          </div>
        </div>

        {/* ── RESULTS ── */}
        {(xrayResult && vitalsResult) && (
          <div style={{ marginTop: '28px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px', flexWrap: 'wrap', gap: '12px' }}>
              <div>
                <h2 style={{ fontSize: '17px', fontWeight: '800', color: '#0f172a', letterSpacing: '-0.4px' }}>Diagnosis Results</h2>
                <p style={{ color: '#94a3b8', fontSize: '11px', marginTop: '2px' }}>{new Date().toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}</p>
              </div>
              {patient.name && (
                <div style={{ background: '#fff', borderRadius: '10px', padding: '10px 14px', border: '1px solid #f0f2f5', textAlign: 'right', boxShadow: '0 1px 3px rgba(0,0,0,0.04)' }}>
                  <p style={{ color: '#94a3b8', fontSize: '9px', fontWeight: '700', letterSpacing: '1px' }}>PATIENT</p>
                  <p style={{ color: '#0f172a', fontWeight: '800', fontSize: '14px', letterSpacing: '-0.3px' }}>{patient.name}</p>
                  <p style={{ color: '#64748b', fontSize: '11px' }}>{[patient.id, patient.age && `Age ${patient.age}`, patient.gender].filter(Boolean).join(' · ')}</p>
                </div>
              )}
            </div>

            <div style={{ display: 'flex', gap: '16px', marginBottom: '16px' }}>
              <ResultCard icon="🫁" title="Chest X-Ray Analysis" result={xrayResult} color="#2563eb" />
              <ResultCard icon="🩸" title="Diabetes Risk Analysis" result={vitalsResult} color="#10b981" />
            </div>

            <div style={{ display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap' }}>
              <div style={{ flex: 1, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: '10px', padding: '10px 14px' }}>
                <p style={{ color: '#92400e', fontSize: '12px' }}><strong>Disclaimer:</strong> AI-assisted support only. Always consult a qualified physician before making clinical decisions.</p>
              </div>
              <button onClick={downloadReport} disabled={reportLoading}
                onMouseEnter={e => !reportLoading && (e.currentTarget.style.opacity = '0.85')}
                onMouseLeave={e => (e.currentTarget.style.opacity = '1')}
                style={{ padding: '10px 20px', border: 'none', borderRadius: '8px', background: reportLoading ? '#f0f2f5' : '#10b981', color: reportLoading ? '#94a3b8' : 'white', fontWeight: '700', fontSize: '12px', cursor: reportLoading ? 'not-allowed' : 'pointer', boxShadow: reportLoading ? 'none' : '0 2px 8px rgba(16,185,129,0.3)', transition: 'all 0.2s', whiteSpace: 'nowrap' }}>
                {reportLoading ? 'Generating...' : 'Download PDF Report'}
              </button>
            </div>

            <HeatmapSection heatmap={xrayHeatmap} color="#2563eb" />
          </div>
        )}

        {brainResult && (
          <div style={{ marginTop: '28px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px', flexWrap: 'wrap', gap: '12px' }}>
              <div>
                <h2 style={{ fontSize: '17px', fontWeight: '800', color: '#0f172a', letterSpacing: '-0.4px' }}>Brain MRI Results</h2>
                <p style={{ color: '#94a3b8', fontSize: '11px', marginTop: '2px' }}>{new Date().toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}</p>
              </div>
              {patient.name && (
                <div style={{ background: '#fff', borderRadius: '10px', padding: '10px 14px', border: '1px solid #f0f2f5', textAlign: 'right', boxShadow: '0 1px 3px rgba(0,0,0,0.04)' }}>
                  <p style={{ color: '#94a3b8', fontSize: '9px', fontWeight: '700', letterSpacing: '1px' }}>PATIENT</p>
                  <p style={{ color: '#0f172a', fontWeight: '800', fontSize: '14px', letterSpacing: '-0.3px' }}>{patient.name}</p>
                  <p style={{ color: '#64748b', fontSize: '11px' }}>{[patient.id, patient.age && `Age ${patient.age}`, patient.gender].filter(Boolean).join(' · ')}</p>
                </div>
              )}
            </div>

            <ResultCard icon="🧠" title="Brain Tumor Analysis" result={brainResult} color="#7c3aed" />

            <div style={{ display: 'flex', gap: '12px', alignItems: 'center', marginTop: '16px', flexWrap: 'wrap' }}>
              <div style={{ flex: 1, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: '10px', padding: '10px 14px' }}>
                <p style={{ color: '#92400e', fontSize: '12px' }}><strong>Disclaimer:</strong> AI-assisted support only. Always consult a qualified physician before making clinical decisions.</p>
              </div>
              <button onClick={downloadReport} disabled={reportLoading}
                onMouseEnter={e => !reportLoading && (e.currentTarget.style.opacity = '0.85')}
                onMouseLeave={e => (e.currentTarget.style.opacity = '1')}
                style={{ padding: '10px 20px', border: 'none', borderRadius: '8px', background: reportLoading ? '#f0f2f5' : '#7c3aed', color: reportLoading ? '#94a3b8' : 'white', fontWeight: '700', fontSize: '12px', cursor: reportLoading ? 'not-allowed' : 'pointer', boxShadow: reportLoading ? 'none' : '0 2px 8px rgba(124,58,237,0.3)', transition: 'all 0.2s', whiteSpace: 'nowrap' }}>
                {reportLoading ? 'Generating...' : 'Download PDF Report'}
              </button>
            </div>

            <HeatmapSection heatmap={brainHeatmap} color="#7c3aed" />
          </div>
        )}

        <div style={{ marginTop: '40px', paddingTop: '20px', borderTop: '1px solid #edf0f4', textAlign: 'center' }}>
          <p style={{ color: '#cbd5e1', fontSize: '11px', letterSpacing: '0.3px' }}>MediAI Diagnostics · GLS University Capstone 2025–26 · Integrated MSc(IT)</p>
        </div>
      </div>
    </div>
  )
}