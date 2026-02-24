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
  const handleImage = (e) => { const f = e.target.files[0]; if (!f) return; setImage(f); setPreview(URL.createObjectURL(f)); setError(null) }

  const savePrediction = async (data) => {
    try { await axios.post(`${API}/save-prediction`, data); setSavedMsg('Saved to database'); setTimeout(() => setSavedMsg(null), 4000) }
    catch { console.error('Failed to save') }
  }

  const analyze = async () => {
    setError(null); setSavedMsg(null)
    if (!image) { setError('Please upload an image first'); return }
    if (module === 'xray' && (!vitals.glucose || !vitals.bmi || !vitals.age)) { setError('Please fill Glucose, BMI and Age'); return }
    setLoading(true); setXrayResult(null); setVitalsResult(null); setBrainResult(null); setXrayHeatmap(null); setBrainHeatmap(null)
    try {
      const imgForm = new FormData(); imgForm.append('image', image)
      if (module === 'xray') {
        const vForm = new FormData(); Object.keys(vitals).forEach(k => vForm.append(k, vitals[k] || 0))
        const [xr, vr] = await Promise.all([axios.post(`${API}/predict-xray`, imgForm), axios.post(`${API}/predict-vitals`, vForm)])
        setXrayResult(xr.data); setVitalsResult(vr.data)
        await savePrediction({ patient_id: patient.id || 'N/A', patient_name: patient.name || 'Unknown', patient_age: patient.age || 'N/A', patient_gender: patient.gender, module: 'xray', diagnosis: xr.data.diagnosis, risk_score: xr.data.risk_score, probabilities: xr.data.probabilities, vitals_diagnosis: vr.data.diagnosis, vitals_risk_score: vr.data.risk_score, vitals_probabilities: vr.data.probabilities, saved_by: user?.username || 'Unknown' })
      } else {
        const br = await axios.post(`${API}/predict-brain`, imgForm)
        if (br.data.status === 'error') setError(br.data.message)
        else { setBrainResult(br.data); await savePrediction({ patient_id: patient.id || 'N/A', patient_name: patient.name || 'Unknown', patient_age: patient.age || 'N/A', patient_gender: patient.gender, module: 'brain', diagnosis: br.data.diagnosis, risk_score: br.data.risk_score, probabilities: br.data.probabilities, saved_by: user?.username || 'Unknown' }) }
      }
    } catch { setError('Cannot connect to backend. Make sure server is running.') }
    setLoading(false)
  }

  const generateHeatmaps = async () => {
    if (!image) return; setGradcamLoading(true)
    try {
      const f = new FormData(); f.append('image', image)
      if (module === 'xray') { const r = await axios.post(`${API}/gradcam-xray`, f); setXrayHeatmap(r.data.heatmap) }
      else { const r = await axios.post(`${API}/gradcam-brain`, f); setBrainHeatmap(r.data.heatmap) }
    } catch { setError('Failed to generate heatmap') }
    setGradcamLoading(false)
  }

  const downloadReport = async () => {
    if (!image) return; setReportLoading(true)
    try {
      const form = new FormData()
      form.append('image', image); form.append('module', module); form.append('patient_name', patient.name || ''); form.append('patient_id', patient.id || ''); form.append('patient_age', patient.age || ''); form.append('patient_gender', patient.gender || 'Male')
      if (module === 'xray' && xrayResult && vitalsResult) { form.append('xray_diagnosis', xrayResult.diagnosis); form.append('xray_risk_score', xrayResult.risk_score); form.append('xray_prob_normal', xrayResult.probabilities['Normal']); form.append('xray_prob_pneumonia', xrayResult.probabilities['Pneumonia']); form.append('vitals_diagnosis', vitalsResult.diagnosis); form.append('vitals_risk_score', vitalsResult.risk_score); form.append('vitals_prob_no_diabetes', vitalsResult.probabilities['No Diabetes']); form.append('vitals_prob_diabetes', vitalsResult.probabilities['Diabetes']); form.append('heatmap', xrayHeatmap || '') }
      if (module === 'brain' && brainResult) { form.append('brain_diagnosis', brainResult.diagnosis); form.append('brain_risk_score', brainResult.risk_score); form.append('brain_prob_glioma', brainResult.probabilities['Glioma']); form.append('brain_prob_meningioma', brainResult.probabilities['Meningioma']); form.append('brain_prob_no_tumor', brainResult.probabilities['No Tumor']); form.append('brain_prob_pituitary', brainResult.probabilities['Pituitary']); form.append('heatmap', brainHeatmap || '') }
      const res = await axios.post(`${API}/generate-report`, form, { responseType: 'blob' })
      const url = window.URL.createObjectURL(new Blob([res.data])); const a = document.createElement('a'); a.href = url; a.setAttribute('download', `MediAI_${patient.name || 'Report'}.pdf`); document.body.appendChild(a); a.click(); a.remove()
    } catch { setError('Failed to generate PDF') }
    setReportLoading(false)
  }

  const isXray = module === 'xray'
  const accent = isXray ? '#2563eb' : '#7c3aed'
  const riskColor = s => s >= 70 ? '#ef4444' : s >= 40 ? '#f59e0b' : '#10b981'
  const riskBg = s => s >= 70 ? '#fef2f2' : s >= 40 ? '#fefce8' : '#f0fdf4'
  const riskBorder = s => s >= 70 ? '#fecaca' : s >= 40 ? '#fef08a' : '#bbf7d0'
  const riskLabel = s => s >= 70 ? 'High Risk' : s >= 40 ? 'Moderate Risk' : 'Low Risk'

  // ─── SHARED STYLES ───────────────────────────────────────────────────────
  const G = {
    page: { minHeight: '100vh', background: '#f8fafc', fontFamily: "'Helvetica Neue', Arial, sans-serif", color: '#0f172a' },
    nav: { height: '60px', background: '#fff', borderBottom: '1px solid #e2e8f0', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 32px', position: 'sticky', top: 0, zIndex: 100 },
    banner: { background: isXray ? 'linear-gradient(135deg,#1e3a8a,#3b82f6)' : 'linear-gradient(135deg,#4c1d95,#8b5cf6)', padding: '28px 32px', color: '#fff' },
    body: { maxWidth: '1160px', margin: '0 auto', padding: '28px 32px' },
    card: { background: '#fff', borderRadius: '14px', border: '1px solid #e2e8f0', padding: '22px', marginBottom: '18px', boxShadow: '0 1px 4px rgba(0,0,0,0.04)' },
    inp: { width: '100%', padding: '10px 13px', border: '1.5px solid #e2e8f0', borderRadius: '8px', fontSize: '13px', color: '#0f172a', background: '#f8fafc', boxSizing: 'border-box', outline: 'none', fontFamily: 'inherit', transition: 'border-color .2s,box-shadow .2s', marginTop: '5px' },
    lbl: { fontSize: '11px', fontWeight: '700', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.6px', display: 'block' },
    chip: { background: 'rgba(255,255,255,0.18)', border: '1px solid rgba(255,255,255,0.25)', borderRadius: '10px', padding: '10px 18px', backdropFilter: 'blur(6px)' },
  }

  // ─── RESULT CARD ─────────────────────────────────────────────────────────
  const ResultCard = ({ icon, title, result, color }) => (
    <div style={{ ...G.card, flex: 1, marginBottom: 0, borderTop: `3px solid ${color}`, boxShadow: '0 2px 12px rgba(0,0,0,0.06)' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '18px' }}>
        <div style={{ width: '38px', height: '38px', borderRadius: '10px', background: `${color}15`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '18px' }}>{icon}</div>
        <div>
          <p style={{ fontWeight: '700', fontSize: '14px', color: '#0f172a' }}>{title}</p>
          <p style={{ fontSize: '11px', color: '#94a3b8' }}>AI Analysis Complete</p>
        </div>
      </div>

      <div style={{ background: riskBg(result.risk_score), border: `1px solid ${riskBorder(result.risk_score)}`, borderRadius: '12px', padding: '16px 18px', marginBottom: '16px' }}>
        <p style={{ fontSize: '10px', fontWeight: '700', color: '#94a3b8', letterSpacing: '1.2px', marginBottom: '6px' }}>PRIMARY DIAGNOSIS</p>
        <p style={{ fontSize: '22px', fontWeight: '800', color: '#0f172a', letterSpacing: '-0.5px', lineHeight: 1.1 }}>{result.diagnosis}</p>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '10px' }}>
          <span style={{ background: riskColor(result.risk_score), color: '#fff', fontSize: '10px', fontWeight: '700', padding: '3px 10px', borderRadius: '20px' }}>{riskLabel(result.risk_score)}</span>
          <span style={{ color: '#94a3b8', fontSize: '12px' }}>Score: <b style={{ color: riskColor(result.risk_score) }}>{result.risk_score}</b>/100</span>
        </div>
      </div>

      <div style={{ background: '#f1f5f9', borderRadius: '6px', height: '6px', overflow: 'hidden', marginBottom: '18px' }}>
        <div style={{ width: `${result.risk_score}%`, height: '100%', background: `linear-gradient(90deg,${color},${riskColor(result.risk_score)})`, borderRadius: '6px', transition: 'width 1.4s cubic-bezier(.4,0,.2,1)' }} />
      </div>

      <p style={{ fontSize: '10px', fontWeight: '700', color: '#94a3b8', letterSpacing: '1.2px', marginBottom: '12px' }}>PROBABILITY BREAKDOWN</p>
      {Object.entries(result.probabilities).map(([d, p]) => (
        <div key={d} style={{ marginBottom: '10px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
            <span style={{ fontSize: '12px', color: '#475569', fontWeight: '500' }}>{d}</span>
            <span style={{ fontSize: '12px', fontWeight: '800', color: '#0f172a' }}>{p}%</span>
          </div>
          <div style={{ background: '#f1f5f9', borderRadius: '4px', height: '5px', overflow: 'hidden' }}>
            <div style={{ width: `${p}%`, height: '100%', background: color, borderRadius: '4px', transition: 'width 1.4s cubic-bezier(.4,0,.2,1)' }} />
          </div>
        </div>
      ))}
    </div>
  )

  // ─── HEATMAP ─────────────────────────────────────────────────────────────
  const HeatmapSection = ({ heatmap, color }) => (
    <div style={{ ...G.card, marginTop: '20px', borderColor: `${color}30` }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
        <div>
          <p style={{ fontWeight: '700', fontSize: '14px', color: '#0f172a' }}>Grad-CAM Heatmap</p>
          <p style={{ fontSize: '12px', color: '#94a3b8', marginTop: '2px' }}>AI attention visualization</p>
        </div>
        <button onClick={generateHeatmaps} disabled={gradcamLoading}
          style={{ padding: '9px 18px', border: 'none', borderRadius: '8px', background: gradcamLoading ? '#f1f5f9' : color, color: gradcamLoading ? '#94a3b8' : '#fff', fontWeight: '600', fontSize: '12px', cursor: gradcamLoading ? 'not-allowed' : 'pointer', transition: 'opacity .2s' }}>
          {gradcamLoading ? 'Generating...' : 'Generate Heatmap'}
        </button>
      </div>
      {!heatmap && !gradcamLoading && (
        <div style={{ background: '#f8fafc', borderRadius: '10px', padding: '32px', textAlign: 'center', border: '2px dashed #e2e8f0' }}>
          <p style={{ color: '#475569', fontWeight: '600', fontSize: '13px' }}>Click "Generate Heatmap" to visualize AI attention regions</p>
          <p style={{ color: '#94a3b8', fontSize: '12px', marginTop: '4px' }}>Red/yellow = high attention · Blue = low attention</p>
        </div>
      )}
      {gradcamLoading && <div style={{ background: '#f8fafc', borderRadius: '10px', padding: '32px', textAlign: 'center' }}><p style={{ color: '#64748b', fontWeight: '600', fontSize: '13px' }}>Computing heatmap...</p></div>}
      {heatmap && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '12px' }}>
            {[['Original Scan', preview], ['AI Attention Map', `data:image/jpeg;base64,${heatmap}`]].map(([lbl, src]) => (
              <div key={lbl} style={{ textAlign: 'center' }}>
                <p style={{ fontSize: '10px', fontWeight: '700', color: '#94a3b8', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '1px' }}>{lbl}</p>
                <img src={src} alt={lbl} style={{ width: '100%', borderRadius: '10px', objectFit: 'contain', maxHeight: '220px', background: '#000' }} />
              </div>
            ))}
          </div>
          <div style={{ background: '#f8fafc', borderRadius: '8px', padding: '10px 14px', border: '1px solid #e2e8f0' }}>
            <p style={{ fontSize: '12px', color: '#64748b' }}>Red/yellow regions = AI detected abnormality · Blue regions = lower diagnostic significance</p>
          </div>
        </>
      )}
    </div>
  )

  // ─── MAIN RENDER ─────────────────────────────────────────────────────────
  return (
    <div style={G.page}>

      {/* ── NAVBAR ── */}
      <nav style={G.nav}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{ width: '32px', height: '32px', borderRadius: '9px', background: 'linear-gradient(135deg,#2563eb,#0ea5e9)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff', fontWeight: '900', fontSize: '15px' }}>+</div>
          <div>
            <p style={{ fontSize: '14px', fontWeight: '800', color: '#0f172a', lineHeight: 1, letterSpacing: '-0.3px' }}>MediAI Diagnostics</p>
            <p style={{ fontSize: '10px', color: '#94a3b8', letterSpacing: '0.3px' }}>AI-Powered Clinical Decision Support</p>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {[['xray', 'Chest X-Ray'], ['brain', 'Brain MRI']].map(([m, lbl]) => (
            <button key={m} onClick={() => switchModule(m)} style={{ padding: '7px 16px', borderRadius: '8px', border: module === m ? 'none' : '1.5px solid #e2e8f0', fontWeight: '600', fontSize: '13px', cursor: 'pointer', background: module === m ? (m === 'xray' ? '#2563eb' : '#7c3aed') : '#fff', color: module === m ? '#fff' : '#64748b', transition: 'all .2s' }}>{lbl}</button>
          ))}

          <div style={{ width: '1px', height: '22px', background: '#e2e8f0', margin: '0 4px' }} />

          {user && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '7px', padding: '5px 11px', background: '#f0fdf4', border: '1px solid #bbf7d0', borderRadius: '8px' }}>
              <div style={{ width: '22px', height: '22px', borderRadius: '50%', background: '#10b981', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff', fontSize: '11px', fontWeight: '800' }}>{user.username?.[0]?.toUpperCase()}</div>
              <span style={{ color: '#059669', fontWeight: '700', fontSize: '13px' }}>{user.username}</span>
            </div>
          )}

          <button onClick={() => { reset(); setScreen('welcome'); setUser(null) }}
            onMouseEnter={e => e.currentTarget.style.background = '#fef2f2'}
            onMouseLeave={e => e.currentTarget.style.background = '#fff'}
            style={{ padding: '7px 14px', borderRadius: '8px', border: '1.5px solid #fecaca', fontWeight: '600', fontSize: '13px', cursor: 'pointer', background: '#fff', color: '#dc2626', transition: 'background .2s' }}>
            Sign Out
          </button>
        </div>
      </nav>

      {/* ── BANNER ── */}
      <div style={G.banner}>
        <div style={{ maxWidth: '1160px', margin: '0 auto', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '20px', flexWrap: 'wrap' }}>
          <div>
            <p style={{ fontSize: '11px', fontWeight: '700', letterSpacing: '2px', textTransform: 'uppercase', color: 'rgba(255,255,255,0.6)', marginBottom: '6px' }}>{isXray ? 'Chest X-Ray + Diabetes' : 'Brain MRI'}</p>
            <h2 style={{ fontSize: '22px', fontWeight: '800', color: '#fff', letterSpacing: '-0.5px', marginBottom: '4px' }}>
              {isXray ? 'Pneumonia Detection & Diabetes Risk Assessment' : 'Brain Tumor Detection & Classification'}
            </h2>
            <p style={{ color: 'rgba(255,255,255,0.6)', fontSize: '13px' }}>
              {isXray ? 'Powered by ResNet-50 with Grad-CAM explainability' : 'Powered by EfficientNet-B3 with Grad-CAM explainability'}
            </p>
          </div>
          <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
            {(isXray
              ? [['98%', 'X-Ray Acc.'], ['78%', 'Diabetes Acc.'], ['ResNet-50', 'Model'], ['Grad-CAM', 'XAI'], ['PDF', 'Export']]
              : [['94.75%', 'MRI Acc.'], ['4', 'Tumor Types'], ['EfficientNet-B3', 'Model'], ['Grad-CAM', 'XAI'], ['PDF', 'Export']]
            ).map(([v, l]) => (
              <div key={l} style={G.chip}>
                <p style={{ color: '#fff', fontWeight: '800', fontSize: '15px', lineHeight: 1 }}>{v}</p>
                <p style={{ color: 'rgba(255,255,255,0.55)', fontSize: '10px', marginTop: '3px', textTransform: 'uppercase', letterSpacing: '0.8px' }}>{l}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── BODY ── */}
      <div style={G.body}>

        {/* Alerts */}
        {error && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', background: '#fef2f2', border: '1px solid #fecaca', borderRadius: '10px', padding: '12px 16px', marginBottom: '20px' }}>
            <span style={{ color: '#dc2626', fontSize: '16px', fontWeight: '700' }}>⚠</span>
            <p style={{ color: '#dc2626', fontSize: '13px', flex: 1 }}>{error}</p>
            <button onClick={() => setError(null)} style={{ background: 'none', border: 'none', color: '#dc2626', fontSize: '18px', cursor: 'pointer', lineHeight: 1 }}>×</button>
          </div>
        )}
        {savedMsg && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', background: '#f0fdf4', border: '1px solid #bbf7d0', borderRadius: '10px', padding: '12px 16px', marginBottom: '20px' }}>
            <span style={{ color: '#10b981' }}>✓</span>
            <p style={{ color: '#059669', fontSize: '13px' }}>{savedMsg}</p>
          </div>
        )}

        {/* Two column grid */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>

          {/* ── LEFT ── */}
          <div>
            {/* Patient */}
            <div style={G.card}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', paddingBottom: '14px', marginBottom: '16px', borderBottom: '1px solid #f1f5f9' }}>
                <div style={{ width: '34px', height: '34px', borderRadius: '9px', background: '#eff6ff', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '16px' }}>👤</div>
                <div>
                  <p style={{ fontWeight: '700', fontSize: '14px', color: '#0f172a' }}>Patient Information</p>
                  <p style={{ fontSize: '11px', color: '#94a3b8' }}>Enter patient details</p>
                </div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                <div style={{ gridColumn: '1/-1' }}>
                  <label style={G.lbl}>Full Name</label>
                  <input style={G.inp} value={patient.name} onChange={e => setPatient({ ...patient, name: e.target.value })}
                    onFocus={e => { e.target.style.borderColor = accent; e.target.style.boxShadow = `0 0 0 3px ${accent}20` }}
                    onBlur={e => { e.target.style.borderColor = '#e2e8f0'; e.target.style.boxShadow = 'none' }} />
                </div>
                <div>
                  <label style={G.lbl}>Patient ID</label>
                  <input style={G.inp} placeholder="PT-001" value={patient.id} onChange={e => setPatient({ ...patient, id: e.target.value })}
                    onFocus={e => { e.target.style.borderColor = accent; e.target.style.boxShadow = `0 0 0 3px ${accent}20` }}
                    onBlur={e => { e.target.style.borderColor = '#e2e8f0'; e.target.style.boxShadow = 'none' }} />
                </div>
                <div>
                  <label style={G.lbl}>Age</label>
                  <input style={G.inp} type="number" value={patient.age} onChange={e => setPatient({ ...patient, age: e.target.value })}
                    onFocus={e => { e.target.style.borderColor = accent; e.target.style.boxShadow = `0 0 0 3px ${accent}20` }}
                    onBlur={e => { e.target.style.borderColor = '#e2e8f0'; e.target.style.boxShadow = 'none' }} />
                </div>
                <div>
                  <label style={G.lbl}>Gender</label>
                  <select style={G.inp} value={patient.gender} onChange={e => setPatient({ ...patient, gender: e.target.value })}>
                    <option>Male</option><option>Female</option><option>Other</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Upload */}
            <div style={G.card}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', paddingBottom: '14px', marginBottom: '16px', borderBottom: '1px solid #f1f5f9' }}>
                <div style={{ width: '34px', height: '34px', borderRadius: '9px', background: isXray ? '#eff6ff' : '#f5f3ff', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '16px' }}>{isXray ? '🫁' : '🧠'}</div>
                <div>
                  <p style={{ fontWeight: '700', fontSize: '14px', color: '#0f172a' }}>{isXray ? 'Chest X-Ray Image' : 'Brain MRI Scan'}</p>
                  <p style={{ fontSize: '11px', color: '#94a3b8' }}>Upload your medical scan</p>
                </div>
              </div>
              <label style={{ display: 'block', border: `2px dashed ${preview ? accent : '#e2e8f0'}`, borderRadius: '12px', padding: '22px', textAlign: 'center', cursor: 'pointer', background: preview ? `${accent}08` : '#f8fafc', transition: 'all .2s' }}>
                <input type="file" accept="image/*" onChange={handleImage} style={{ display: 'none' }} />
                {preview
                  ? <img src={preview} alt="preview" style={{ maxWidth: '100%', maxHeight: '160px', borderRadius: '8px', objectFit: 'contain' }} />
                  : <>
                    <div style={{ width: '40px', height: '40px', borderRadius: '10px', background: '#f1f5f9', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '18px', margin: '0 auto 10px' }}>📁</div>
                    <p style={{ color: '#475569', fontSize: '13px', fontWeight: '600' }}>Click to upload image</p>
                    <p style={{ color: '#94a3b8', fontSize: '11px', marginTop: '3px' }}>PNG, JPG, JPEG · Max 10MB</p>
                  </>
                }
              </label>
              {!isXray && <div style={{ background: '#faf5ff', border: '1px solid #e9d5ff', borderRadius: '8px', padding: '8px 12px', marginTop: '10px' }}><p style={{ color: '#7c3aed', fontSize: '11px', fontWeight: '600' }}>Only Brain MRI scans accepted</p></div>}
            </div>
          </div>

          {/* ── RIGHT ── */}
          <div>
            {isXray ? (
              <div style={G.card}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', paddingBottom: '14px', marginBottom: '16px', borderBottom: '1px solid #f1f5f9' }}>
                  <div style={{ width: '34px', height: '34px', borderRadius: '9px', background: '#f0fdf4', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '16px' }}>💉</div>
                  <div>
                    <p style={{ fontWeight: '700', fontSize: '14px', color: '#0f172a' }}>Patient Vitals</p>
                    <p style={{ fontSize: '11px', color: '#94a3b8' }}>Lab values for diabetes risk</p>
                  </div>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '11px' }}>
                  {[['pregnancies','Pregnancies',''],['glucose','Glucose (mg/dL)','*'],['blood_pressure','Blood Pressure',''],['skin_thickness','Skin Thickness',''],['insulin','Insulin (μU/mL)',''],['bmi','BMI (kg/m²)','*'],['diabetes_pedigree','Pedigree Fn.',''],['age','Age (Years)','*']].map(([key, label, req]) => (
                    <div key={key}>
                      <label style={G.lbl}>{label} {req && <span style={{ color: '#ef4444' }}>{req}</span>}</label>
                      <input type="number" style={G.inp} value={vitals[key]} onChange={e => setVitals({ ...vitals, [key]: e.target.value })}
                        onFocus={e => { e.target.style.borderColor = '#10b981'; e.target.style.boxShadow = '0 0 0 3px #10b98118' }}
                        onBlur={e => { e.target.style.borderColor = '#e2e8f0'; e.target.style.boxShadow = 'none' }} />
                    </div>
                  ))}
                </div>
                <p style={{ fontSize: '10px', color: '#94a3b8', marginTop: '10px' }}>* Required fields</p>
              </div>
            ) : (
              <div style={G.card}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', paddingBottom: '14px', marginBottom: '16px', borderBottom: '1px solid #f1f5f9' }}>
                  <div style={{ width: '34px', height: '34px', borderRadius: '9px', background: '#f5f3ff', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '16px' }}>🔬</div>
                  <div>
                    <p style={{ fontWeight: '700', fontSize: '14px', color: '#0f172a' }}>Detectable Conditions</p>
                    <p style={{ fontSize: '11px', color: '#94a3b8' }}>4 tumor types supported</p>
                  </div>
                </div>
                {[
                  ['Glioma', 'Most common malignant brain tumor', '#ef4444', '#fef2f2', '#fecaca'],
                  ['Meningioma', 'Usually benign, arises from meninges', '#f59e0b', '#fefce8', '#fef08a'],
                  ['Pituitary', 'Affects the pituitary gland', '#8b5cf6', '#f5f3ff', '#ddd6fe'],
                  ['No Tumor', 'No abnormality detected', '#10b981', '#f0fdf4', '#bbf7d0'],
                ].map(([name, desc, color, bg, border]) => (
                  <div key={name} style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '12px 14px', background: bg, border: `1px solid ${border}`, borderRadius: '10px', marginBottom: '8px' }}>
                    <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: color, flexShrink: 0 }} />
                    <div>
                      <p style={{ fontWeight: '700', fontSize: '13px', color: '#0f172a' }}>{name}</p>
                      <p style={{ fontSize: '11px', color: '#64748b', marginTop: '1px' }}>{desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Analyze btn */}
            <button onClick={analyze} disabled={loading}
              onMouseEnter={e => !loading && (e.currentTarget.style.opacity = '0.88')}
              onMouseLeave={e => (e.currentTarget.style.opacity = '1')}
              style={{ width: '100%', padding: '14px', border: 'none', borderRadius: '10px', fontSize: '14px', fontWeight: '700', cursor: loading ? 'not-allowed' : 'pointer', background: loading ? '#f1f5f9' : isXray ? 'linear-gradient(135deg,#1e40af,#3b82f6)' : 'linear-gradient(135deg,#5b21b6,#8b5cf6)', color: loading ? '#94a3b8' : '#fff', boxShadow: loading ? 'none' : `0 4px 14px ${accent}35`, transition: 'all .2s' }}>
              {loading ? 'Analyzing...' : 'Run AI Diagnosis'}
            </button>
          </div>
        </div>

        {/* ── XRAY RESULTS ── */}
        {xrayResult && vitalsResult && (
          <div style={{ marginTop: '32px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '18px', flexWrap: 'wrap', gap: '12px' }}>
              <div>
                <h2 style={{ fontSize: '18px', fontWeight: '800', color: '#0f172a', letterSpacing: '-0.4px' }}>Diagnosis Results</h2>
                <p style={{ fontSize: '12px', color: '#94a3b8', marginTop: '2px' }}>{new Date().toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}</p>
              </div>
              {patient.name && (
                <div style={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: '10px', padding: '10px 16px', textAlign: 'right' }}>
                  <p style={{ fontSize: '9px', fontWeight: '700', color: '#94a3b8', letterSpacing: '1px' }}>PATIENT</p>
                  <p style={{ fontSize: '15px', fontWeight: '800', color: '#0f172a', letterSpacing: '-0.3px' }}>{patient.name}</p>
                  <p style={{ fontSize: '11px', color: '#64748b' }}>{[patient.id, patient.age && `Age ${patient.age}`, patient.gender].filter(Boolean).join(' · ')}</p>
                </div>
              )}
            </div>

            <div style={{ display: 'flex', gap: '18px', marginBottom: '16px' }}>
              <ResultCard icon="🫁" title="Chest X-Ray Analysis" result={xrayResult} color="#2563eb" />
              <ResultCard icon="🩸" title="Diabetes Risk Analysis" result={vitalsResult} color="#10b981" />
            </div>

            <div style={{ display: 'flex', gap: '14px', alignItems: 'center', flexWrap: 'wrap' }}>
              <div style={{ flex: 1, background: '#fefce8', border: '1px solid #fef08a', borderRadius: '10px', padding: '10px 14px' }}>
                <p style={{ color: '#854d0e', fontSize: '12px' }}><b>Disclaimer:</b> AI-assisted decision support only. Always consult a qualified physician.</p>
              </div>
              <button onClick={downloadReport} disabled={reportLoading}
                onMouseEnter={e => !reportLoading && (e.currentTarget.style.opacity = '0.85')}
                onMouseLeave={e => (e.currentTarget.style.opacity = '1')}
                style={{ padding: '10px 20px', border: 'none', borderRadius: '8px', background: reportLoading ? '#f1f5f9' : '#10b981', color: reportLoading ? '#94a3b8' : '#fff', fontWeight: '700', fontSize: '13px', cursor: reportLoading ? 'not-allowed' : 'pointer', whiteSpace: 'nowrap', boxShadow: reportLoading ? 'none' : '0 2px 8px #10b98130', transition: 'opacity .2s' }}>
                {reportLoading ? 'Generating...' : 'Download PDF Report'}
              </button>
            </div>
            <HeatmapSection heatmap={xrayHeatmap} color="#2563eb" />
          </div>
        )}

        {/* ── BRAIN RESULTS ── */}
        {brainResult && (
          <div style={{ marginTop: '32px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '18px', flexWrap: 'wrap', gap: '12px' }}>
              <div>
                <h2 style={{ fontSize: '18px', fontWeight: '800', color: '#0f172a', letterSpacing: '-0.4px' }}>Brain MRI Results</h2>
                <p style={{ fontSize: '12px', color: '#94a3b8', marginTop: '2px' }}>{new Date().toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}</p>
              </div>
              {patient.name && (
                <div style={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: '10px', padding: '10px 16px', textAlign: 'right' }}>
                  <p style={{ fontSize: '9px', fontWeight: '700', color: '#94a3b8', letterSpacing: '1px' }}>PATIENT</p>
                  <p style={{ fontSize: '15px', fontWeight: '800', color: '#0f172a', letterSpacing: '-0.3px' }}>{patient.name}</p>
                  <p style={{ fontSize: '11px', color: '#64748b' }}>{[patient.id, patient.age && `Age ${patient.age}`, patient.gender].filter(Boolean).join(' · ')}</p>
                </div>
              )}
            </div>
            <ResultCard icon="🧠" title="Brain Tumor Analysis" result={brainResult} color="#7c3aed" />
            <div style={{ display: 'flex', gap: '14px', alignItems: 'center', marginTop: '16px', flexWrap: 'wrap' }}>
              <div style={{ flex: 1, background: '#fefce8', border: '1px solid #fef08a', borderRadius: '10px', padding: '10px 14px' }}>
                <p style={{ color: '#854d0e', fontSize: '12px' }}><b>Disclaimer:</b> AI-assisted decision support only. Always consult a qualified physician.</p>
              </div>
              <button onClick={downloadReport} disabled={reportLoading}
                onMouseEnter={e => !reportLoading && (e.currentTarget.style.opacity = '0.85')}
                onMouseLeave={e => (e.currentTarget.style.opacity = '1')}
                style={{ padding: '10px 20px', border: 'none', borderRadius: '8px', background: reportLoading ? '#f1f5f9' : '#7c3aed', color: reportLoading ? '#94a3b8' : '#fff', fontWeight: '700', fontSize: '13px', cursor: reportLoading ? 'not-allowed' : 'pointer', whiteSpace: 'nowrap', boxShadow: reportLoading ? 'none' : '0 2px 8px #7c3aed30', transition: 'opacity .2s' }}>
                {reportLoading ? 'Generating...' : 'Download PDF Report'}
              </button>
            </div>
            <HeatmapSection heatmap={brainHeatmap} color="#7c3aed" />
          </div>
        )}

        <div style={{ marginTop: '48px', paddingTop: '20px', borderTop: '1px solid #e2e8f0', textAlign: 'center' }}>
          <p style={{ color: '#cbd5e1', fontSize: '11px', letterSpacing: '0.5px' }}>MediAI Diagnostics · GLS University Capstone 2025–26 · Integrated MSc(IT)</p>
        </div>
      </div>
    </div>
  )
}