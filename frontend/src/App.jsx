import { useState } from 'react'
import axios from 'axios'
import './App.css'

const API = 'http://127.0.0.1:8000'

export default function App() {
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
  const [error, setError] = useState(null)

  const reset = () => {
    setImage(null); setPreview(null); setError(null)
    setXrayResult(null); setVitalsResult(null); setBrainResult(null)
    setXrayHeatmap(null); setBrainHeatmap(null)
    setVitals({ pregnancies: '', glucose: '', blood_pressure: '',
      skin_thickness: '', insulin: '', bmi: '', diabetes_pedigree: '', age: '' })
    setPatient({ name: '', age: '', gender: 'Male', id: '' })
  }

  const switchModule = (m) => { setModule(m); reset() }

  const handleImage = (e) => {
    const file = e.target.files[0]
    if (!file) return
    setImage(file); setPreview(URL.createObjectURL(file)); setError(null)
  }

  const analyze = async () => {
    setError(null)
    if (!image) { setError('Please upload an image first!'); return }
    if (module === 'xray' && (!vitals.glucose || !vitals.bmi || !vitals.age)) {
      setError('Please fill in Glucose, BMI and Age!'); return
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
      } else {
        const br = await axios.post(`${API}/predict-brain`, imgForm)
        if (br.data.status === 'error') setError(br.data.message)
        else setBrainResult(br.data)
      }
    } catch { setError('Cannot connect to backend. Make sure server is running!') }
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

  const riskColor = (s) => s >= 70 ? '#dc2626' : s >= 40 ? '#d97706' : '#16a34a'
  const riskBg = (s) => s >= 70 ? '#fef2f2' : s >= 40 ? '#fffbeb' : '#f0fdf4'
  const riskLabel = (s) => s >= 70 ? 'High Risk' : s >= 40 ? 'Medium Risk' : 'Low Risk'
  const riskIcon = (s) => s >= 70 ? '🔴' : s >= 40 ? '🟡' : '🟢'

  const inp = {
    width: '100%', padding: '10px 14px', borderRadius: '8px',
    border: '1.5px solid #e2e8f0', background: 'white',
    color: '#1e293b', fontSize: '14px', marginTop: '5px',
    boxSizing: 'border-box'
  }
  const lbl = {
    fontSize: '12px', fontWeight: '600', color: '#64748b',
    textTransform: 'uppercase', letterSpacing: '0.5px'
  }
  const card = {
    background: 'white', borderRadius: '16px', padding: '24px',
    boxShadow: '0 1px 3px rgba(0,0,0,0.08), 0 4px 16px rgba(0,0,0,0.04)',
    marginBottom: '20px', border: '1px solid #f1f5f9'
  }

  const ResultCard = ({ icon, title, result, color }) => (
    <div style={{ ...card, flex: 1, borderTop: `4px solid ${color}` }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px' }}>
        <span style={{ fontSize: '22px' }}>{icon}</span>
        <h3 style={{ color: '#1e293b', fontSize: '16px', fontWeight: '700' }}>{title}</h3>
      </div>
      <div style={{ background: riskBg(result.risk_score), borderRadius: '12px',
        padding: '16px', marginBottom: '20px',
        border: `1px solid ${riskColor(result.risk_score)}20` }}>
        <p style={{ color: '#64748b', fontSize: '11px', fontWeight: '700',
          letterSpacing: '0.5px', marginBottom: '4px' }}>PRIMARY DIAGNOSIS</p>
        <p style={{ color: '#0f172a', fontSize: '24px', fontWeight: '800' }}>
          {result.diagnosis}
        </p>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginTop: '8px' }}>
          <span style={{ fontSize: '16px' }}>{riskIcon(result.risk_score)}</span>
          <span style={{ color: riskColor(result.risk_score), fontWeight: '700',
            fontSize: '14px' }}>{riskLabel(result.risk_score)}</span>
          <span style={{ color: '#94a3b8', fontSize: '13px' }}>
            — Score: {result.risk_score}/100
          </span>
        </div>
      </div>
      <div style={{ background: '#f8fafc', borderRadius: '8px',
        height: '8px', marginBottom: '20px', overflow: 'hidden' }}>
        <div style={{ width: `${result.risk_score}%`, height: '100%',
          background: `linear-gradient(90deg, ${color}, ${riskColor(result.risk_score)})`,
          transition: 'width 1.2s ease' }} />
      </div>
      <p style={{ color: '#94a3b8', fontSize: '11px', fontWeight: '700',
        letterSpacing: '0.5px', marginBottom: '12px' }}>PROBABILITY BREAKDOWN</p>
      {Object.entries(result.probabilities).map(([d, p]) => (
        <div key={d} style={{ marginBottom: '12px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
            <span style={{ color: '#475569', fontSize: '13px', fontWeight: '500' }}>{d}</span>
            <span style={{ color: '#0f172a', fontWeight: '700', fontSize: '13px' }}>{p}%</span>
          </div>
          <div style={{ background: '#f1f5f9', borderRadius: '6px', height: '7px' }}>
            <div style={{ width: `${p}%`, height: '100%', background: color,
              borderRadius: '6px', transition: 'width 1.2s ease' }} />
          </div>
        </div>
      ))}
    </div>
  )

  const HeatmapSection = ({ heatmap, color, borderColor }) => (
    <div style={{ ...card, border: `1px solid ${borderColor}`, marginTop: '24px' }}>
      <div style={{ display: 'flex', alignItems: 'center',
        justifyContent: 'space-between', marginBottom: '16px' }}>
        <div>
          <h3 style={{ fontSize: '16px', fontWeight: '700', color: '#0f172a' }}>
            🔥 AI Attention Heatmap (Grad-CAM)
          </h3>
          <p style={{ color: '#64748b', fontSize: '13px', marginTop: '4px' }}>
            Visualize which region the AI focused on to make its diagnosis
          </p>
        </div>
        <button onClick={generateHeatmaps} disabled={gradcamLoading} style={{
          padding: '10px 20px', border: 'none', borderRadius: '8px',
          background: gradcamLoading ? '#e2e8f0' : `linear-gradient(135deg, ${color}, ${color}dd)`,
          color: gradcamLoading ? '#94a3b8' : 'white',
          fontWeight: '700', fontSize: '14px',
          cursor: gradcamLoading ? 'not-allowed' : 'pointer',
          boxShadow: gradcamLoading ? 'none' : `0 4px 12px ${color}40`
        }}>
          {gradcamLoading ? '⏳ Generating...' : '🔥 Generate Heatmap'}
        </button>
      </div>

      {!heatmap && !gradcamLoading && (
        <div style={{ background: '#f8fafc', borderRadius: '10px',
          padding: '32px', textAlign: 'center', border: '2px dashed #e2e8f0' }}>
          <div style={{ fontSize: '40px', marginBottom: '10px' }}>🔬</div>
          <p style={{ color: '#64748b', fontWeight: '600' }}>
            Click "Generate Heatmap" to see AI attention visualization
          </p>
          <p style={{ color: '#94a3b8', fontSize: '13px', marginTop: '4px' }}>
            Shows which areas the AI focused on for diagnosis
          </p>
        </div>
      )}

      {gradcamLoading && (
        <div style={{ background: '#f8fafc', borderRadius: '10px',
          padding: '32px', textAlign: 'center' }}>
          <div style={{ fontSize: '40px', marginBottom: '10px' }}>⏳</div>
          <p style={{ color: '#64748b', fontWeight: '600' }}>Generating heatmap...</p>
        </div>
      )}

      {heatmap && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px',
            marginBottom: '16px' }}>
            <div style={{ textAlign: 'center' }}>
              <p style={{ color: '#64748b', fontSize: '12px', fontWeight: '700',
                marginBottom: '10px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                Original Image
              </p>
              <img src={preview} alt="original"
                style={{ width: '100%', borderRadius: '10px',
                  objectFit: 'contain', maxHeight: '280px', background: '#000' }} />
            </div>
            <div style={{ textAlign: 'center' }}>
              <p style={{ color: '#64748b', fontSize: '12px', fontWeight: '700',
                marginBottom: '10px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                AI Attention Heatmap
              </p>
              <img src={`data:image/jpeg;base64,${heatmap}`} alt="heatmap"
                style={{ width: '100%', borderRadius: '10px',
                  objectFit: 'contain', maxHeight: '280px' }} />
            </div>
          </div>
          <div style={{ background: '#f8fafc', borderRadius: '10px', padding: '14px',
            border: '1px solid #e2e8f0' }}>
            <p style={{ color: '#475569', fontSize: '13px', lineHeight: 1.6 }}>
              <strong>📖 How to read this heatmap:</strong> 🔴 <strong>Red/Yellow</strong> areas
              show where the AI detected abnormality and focused most attention.
              🔵 <strong>Blue</strong> areas are less relevant regions.
              The brighter the color, the more the AI focused on that area for its diagnosis.
            </p>
          </div>
        </div>
      )}
    </div>
  )

  const PatientBadge = () => patient.name ? (
    <div style={{ background: 'white', borderRadius: '12px',
      padding: '12px 20px', boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
      border: '1px solid #f1f5f9', textAlign: 'right' }}>
      <p style={{ color: '#94a3b8', fontSize: '11px', fontWeight: '700' }}>PATIENT</p>
      <p style={{ color: '#0f172a', fontWeight: '800', fontSize: '16px' }}>{patient.name}</p>
      <p style={{ color: '#64748b', fontSize: '12px' }}>
        {patient.id && `ID: ${patient.id} • `}
        {patient.age && `Age: ${patient.age} • `}
        {patient.gender}
      </p>
    </div>
  ) : null

  return (
    <div style={{ minHeight: '100vh', background: '#f0f4f8' }}>

      {/* Top Navigation Bar */}
      <nav style={{ background: 'white', borderBottom: '1px solid #e2e8f0',
        padding: '0 32px', display: 'flex', alignItems: 'center',
        justifyContent: 'space-between', height: '64px',
        boxShadow: '0 1px 3px rgba(0,0,0,0.06)', position: 'sticky', top: 0, zIndex: 100 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ background: 'linear-gradient(135deg, #2563eb, #0ea5e9)',
            borderRadius: '10px', padding: '8px', fontSize: '20px' }}>🏥</div>
          <div>
            <h1 style={{ fontSize: '18px', fontWeight: '800', color: '#0f172a', lineHeight: 1 }}>
              MediAI Diagnostics
            </h1>
            <p style={{ fontSize: '11px', color: '#94a3b8' }}>
              AI-Powered Clinical Decision Support
            </p>
          </div>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          {[['xray', '🫁', 'Chest X-Ray'], ['brain', '🧠', 'Brain MRI']].map(([m, icon, label]) => (
            <button key={m} onClick={() => switchModule(m)} style={{
              padding: '8px 20px', borderRadius: '8px', border: 'none',
              cursor: 'pointer', fontWeight: '600', fontSize: '14px',
              background: module === m ? (m === 'brain' ? '#7c3aed' : '#2563eb') : '#f8fafc',
              color: module === m ? 'white' : '#64748b', transition: 'all 0.2s'
            }}>
              {icon} {label}
            </button>
          ))}
          <button onClick={reset} style={{ padding: '8px 16px', borderRadius: '8px',
            border: '1.5px solid #e2e8f0', cursor: 'pointer', fontWeight: '600',
            fontSize: '14px', background: 'white', color: '#64748b' }}>
            ↺ Reset
          </button>
        </div>
      </nav>

      {/* Page Header */}
      <div style={{ background: module === 'brain'
        ? 'linear-gradient(135deg, #4c1d95, #6d28d9)'
        : 'linear-gradient(135deg, #1e3a8a, #2563eb)',
        padding: '28px 32px', color: 'white' }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
          <h2 style={{ fontSize: '26px', fontWeight: '800', marginBottom: '4px' }}>
            {module === 'xray' ? '🫁 Chest X-Ray + Diabetes Analysis' : '🧠 Brain MRI Tumor Detection'}
          </h2>
          <p style={{ opacity: 0.8, fontSize: '14px' }}>
            {module === 'xray'
              ? 'Upload chest X-ray and enter patient vitals for AI-powered pneumonia and diabetes risk assessment'
              : 'Upload brain MRI scan for AI-powered tumor classification and risk assessment'}
          </p>
          <div style={{ display: 'flex', gap: '16px', marginTop: '20px', flexWrap: 'wrap' }}>
            {(module === 'xray' ? [
              ['X-Ray Accuracy', '98%'],
              ['Diabetes Accuracy', '78%'],
              ['Model', 'ResNet-50'],
              ['Explainability', 'Grad-CAM']
            ] : [
              ['MRI Accuracy', '94.75%'],
              ['Tumor Types', '4 Classes'],
              ['Model', 'EfficientNet-B3'],
              ['Explainability', 'Grad-CAM']
            ]).map(([l, v]) => (
              <div key={l} style={{ background: 'rgba(255,255,255,0.15)',
                borderRadius: '10px', padding: '10px 18px' }}>
                <p style={{ opacity: 0.7, fontSize: '11px', fontWeight: '600' }}>{l}</p>
                <p style={{ fontWeight: '800', fontSize: '16px' }}>{v}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '32px' }}>

        {error && (
          <div style={{ background: '#fef2f2', border: '1px solid #fecaca',
            padding: '14px 18px', borderRadius: '10px', marginBottom: '24px',
            color: '#dc2626', fontSize: '14px', fontWeight: '500',
            display: 'flex', alignItems: 'center', gap: '10px' }}>
            ⚠️ {error}
          </div>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>

          {/* Left Column */}
          <div>
            {/* Patient Info */}
            <div style={card}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px',
                marginBottom: '20px', paddingBottom: '16px', borderBottom: '1px solid #f1f5f9' }}>
                <div style={{ background: '#eff6ff', borderRadius: '8px',
                  padding: '8px', fontSize: '18px' }}>👤</div>
                <div>
                  <h3 style={{ fontSize: '15px', fontWeight: '700', color: '#0f172a' }}>
                    Patient Information
                  </h3>
                  <p style={{ fontSize: '12px', color: '#94a3b8' }}>Enter patient details</p>
                </div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' }}>
                <div style={{ gridColumn: '1/-1' }}>
                  <label style={lbl}>Full Name</label>
                  <input style={inp} placeholder="Enter patient full name"
                    value={patient.name}
                    onChange={e => setPatient({...patient, name: e.target.value})} />
                </div>
                <div>
                  <label style={lbl}>Patient ID</label>
                  <input style={inp} placeholder="PT-001"
                    value={patient.id}
                    onChange={e => setPatient({...patient, id: e.target.value})} />
                </div>
                <div>
                  <label style={lbl}>Age</label>
                  <input style={inp} type="number" placeholder="Years"
                    value={patient.age}
                    onChange={e => setPatient({...patient, age: e.target.value})} />
                </div>
                <div style={{ gridColumn: '1/-1' }}>
                  <label style={lbl}>Gender</label>
                  <select style={inp} value={patient.gender}
                    onChange={e => setPatient({...patient, gender: e.target.value})}>
                    <option>Male</option>
                    <option>Female</option>
                    <option>Other</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Image Upload */}
            <div style={card}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px',
                marginBottom: '20px', paddingBottom: '16px', borderBottom: '1px solid #f1f5f9' }}>
                <div style={{ background: module === 'brain' ? '#f5f3ff' : '#eff6ff',
                  borderRadius: '8px', padding: '8px', fontSize: '18px' }}>
                  {module === 'xray' ? '📷' : '🧠'}
                </div>
                <div>
                  <h3 style={{ fontSize: '15px', fontWeight: '700', color: '#0f172a' }}>
                    {module === 'xray' ? 'Chest X-Ray Image' : 'Brain MRI Scan'}
                  </h3>
                  <p style={{ fontSize: '12px', color: '#94a3b8' }}>Upload medical scan image</p>
                </div>
              </div>
              <label style={{ display: 'block', border: '2px dashed #e2e8f0',
                borderRadius: '12px', padding: '28px', textAlign: 'center',
                cursor: 'pointer', background: '#f8fafc', transition: 'all 0.2s' }}>
                <input type="file" accept="image/*" onChange={handleImage}
                  style={{ display: 'none' }} />
                {preview ? (
                  <img src={preview} alt="preview" style={{ maxWidth: '100%',
                    maxHeight: '220px', borderRadius: '10px', objectFit: 'contain' }} />
                ) : (
                  <>
                    <div style={{ fontSize: '40px', marginBottom: '10px' }}>🖼️</div>
                    <p style={{ color: '#475569', fontSize: '14px', fontWeight: '600' }}>
                      Click to upload image
                    </p>
                    <p style={{ color: '#94a3b8', fontSize: '12px', marginTop: '4px' }}>
                      PNG, JPG, JPEG — Max 10MB
                    </p>
                  </>
                )}
              </label>
              {module === 'brain' && (
                <div style={{ background: '#faf5ff', border: '1px solid #e9d5ff',
                  borderRadius: '8px', padding: '10px 14px', marginTop: '12px' }}>
                  <p style={{ color: '#7c3aed', fontSize: '12px', fontWeight: '600' }}>
                    ⚠️ Only Brain MRI scans accepted. Chest X-rays will be rejected automatically.
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Right Column */}
          <div>
            {module === 'xray' && (
              <div style={card}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px',
                  marginBottom: '20px', paddingBottom: '16px', borderBottom: '1px solid #f1f5f9' }}>
                  <div style={{ background: '#f0fdf4', borderRadius: '8px',
                    padding: '8px', fontSize: '18px' }}>💉</div>
                  <div>
                    <h3 style={{ fontSize: '15px', fontWeight: '700', color: '#0f172a' }}>
                      Patient Vitals
                    </h3>
                    <p style={{ fontSize: '12px', color: '#94a3b8' }}>
                      Enter lab values for diabetes risk
                    </p>
                  </div>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' }}>
                  {[
                    ['pregnancies', 'Pregnancies', '0', ''],
                    ['glucose', 'Glucose (mg/dL)', '120', '*'],
                    ['blood_pressure', 'Blood Pressure', '80', ''],
                    ['skin_thickness', 'Skin Thickness', '20', ''],
                    ['insulin', 'Insulin (μU/mL)', '80', ''],
                    ['bmi', 'BMI (kg/m²)', '25.0', '*'],
                    ['diabetes_pedigree', 'Diabetes Pedigree', '0.5', ''],
                    ['age', 'Age (Years)', '30', '*']
                  ].map(([key, label, ph, req]) => (
                    <div key={key}>
                      <label style={lbl}>
                        {label} <span style={{ color: '#ef4444' }}>{req}</span>
                      </label>
                      <input type="number" style={inp} placeholder={ph}
                        value={vitals[key]}
                        onChange={e => setVitals({...vitals, [key]: e.target.value})} />
                    </div>
                  ))}
                </div>
                <p style={{ color: '#94a3b8', fontSize: '11px', marginTop: '12px' }}>
                  * Required fields for diabetes risk prediction
                </p>
              </div>
            )}

            {module === 'brain' && (
              <div style={card}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px',
                  marginBottom: '20px', paddingBottom: '16px', borderBottom: '1px solid #f1f5f9' }}>
                  <div style={{ background: '#f5f3ff', borderRadius: '8px',
                    padding: '8px', fontSize: '18px' }}>ℹ️</div>
                  <div>
                    <h3 style={{ fontSize: '15px', fontWeight: '700', color: '#0f172a' }}>
                      Detectable Conditions
                    </h3>
                    <p style={{ fontSize: '12px', color: '#94a3b8' }}>4 tumor types supported</p>
                  </div>
                </div>
                {[
                  ['🔴', 'Glioma', 'Most common malignant brain tumor. Originates in glial cells.', '#fef2f2', '#dc2626'],
                  ['🟠', 'Meningioma', 'Usually benign. Arises from meninges surrounding the brain.', '#fff7ed', '#ea580c'],
                  ['🟣', 'Pituitary', 'Affects the pituitary gland and hormone production.', '#faf5ff', '#7c3aed'],
                  ['🟢', 'No Tumor', 'No abnormality detected in the MRI scan.', '#f0fdf4', '#16a34a']
                ].map(([icon, name, desc, bg, color]) => (
                  <div key={name} style={{ display: 'flex', gap: '14px', padding: '14px',
                    borderRadius: '10px', marginBottom: '10px',
                    background: bg, border: `1px solid ${color}20` }}>
                    <span style={{ fontSize: '22px' }}>{icon}</span>
                    <div>
                      <p style={{ color: '#0f172a', fontWeight: '700', fontSize: '14px' }}>{name}</p>
                      <p style={{ color: '#64748b', fontSize: '12px', marginTop: '2px' }}>{desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Analyze Button */}
            <button onClick={analyze} disabled={loading} style={{
              width: '100%', padding: '16px', border: 'none', borderRadius: '12px',
              fontSize: '16px', fontWeight: '700', cursor: loading ? 'not-allowed' : 'pointer',
              background: loading ? '#e2e8f0' :
                module === 'brain'
                  ? 'linear-gradient(135deg, #6d28d9, #7c3aed)'
                  : 'linear-gradient(135deg, #1d4ed8, #2563eb)',
              color: loading ? '#94a3b8' : 'white',
              boxShadow: loading ? 'none' : '0 4px 14px rgba(37,99,235,0.35)',
              transition: 'all 0.3s'
            }}>
              {loading ? '⏳ Analyzing Patient Data...' : '🔍 Run AI Diagnosis'}
            </button>
          </div>
        </div>

        {/* X-Ray Results */}
        {xrayResult && vitalsResult && (
          <div style={{ marginTop: '32px' }}>
            <div style={{ display: 'flex', alignItems: 'center',
              justifyContent: 'space-between', marginBottom: '20px' }}>
              <div>
                <h2 style={{ fontSize: '22px', fontWeight: '800', color: '#0f172a' }}>
                  📊 Diagnosis Results
                </h2>
                <p style={{ color: '#64748b', fontSize: '14px' }}>
                  {new Date().toLocaleDateString('en-US', { weekday: 'long',
                    year: 'numeric', month: 'long', day: 'numeric' })}
                </p>
              </div>
              <PatientBadge />
            </div>
            <div style={{ display: 'flex', gap: '20px' }}>
              <ResultCard icon="🫁" title="Chest X-Ray Analysis"
                result={xrayResult} color="#2563eb" />
              <ResultCard icon="🩸" title="Diabetes Risk Analysis"
                result={vitalsResult} color="#059669" />
            </div>
            <div style={{ background: '#fffbeb', border: '1px solid #fde68a',
              borderRadius: '10px', padding: '12px 18px', marginTop: '16px',
              display: 'flex', alignItems: 'center', gap: '10px' }}>
              <span>⚠️</span>
              <p style={{ color: '#92400e', fontSize: '13px' }}>
                <strong>Medical Disclaimer:</strong> This AI-assisted diagnosis is for
                clinical decision support only. Always consult a qualified medical
                professional before making any clinical decisions.
              </p>
            </div>
            <HeatmapSection heatmap={xrayHeatmap} color="#2563eb" borderColor="#bfdbfe" />
          </div>
        )}

        {/* Brain Results */}
        {brainResult && (
          <div style={{ marginTop: '32px' }}>
            <div style={{ display: 'flex', alignItems: 'center',
              justifyContent: 'space-between', marginBottom: '20px' }}>
              <div>
                <h2 style={{ fontSize: '22px', fontWeight: '800', color: '#0f172a' }}>
                  📊 Brain MRI Results
                </h2>
                <p style={{ color: '#64748b', fontSize: '14px' }}>
                  {new Date().toLocaleDateString('en-US', { weekday: 'long',
                    year: 'numeric', month: 'long', day: 'numeric' })}
                </p>
              </div>
              <PatientBadge />
            </div>
            <ResultCard icon="🧠" title="Brain Tumor Analysis"
              result={brainResult} color="#7c3aed" />
            <div style={{ background: '#fffbeb', border: '1px solid #fde68a',
              borderRadius: '10px', padding: '12px 18px', marginTop: '16px',
              display: 'flex', alignItems: 'center', gap: '10px' }}>
              <span>⚠️</span>
              <p style={{ color: '#92400e', fontSize: '13px' }}>
                <strong>Medical Disclaimer:</strong> This AI-assisted diagnosis is for
                clinical decision support only. Always consult a qualified medical
                professional before making any clinical decisions.
              </p>
            </div>
            <HeatmapSection heatmap={brainHeatmap} color="#7c3aed" borderColor="#ddd6fe" />
          </div>
        )}

        {/* Footer */}
        <div style={{ marginTop: '48px', paddingTop: '24px',
          borderTop: '1px solid #e2e8f0', textAlign: 'center' }}>
          <p style={{ color: '#cbd5e1', fontSize: '13px' }}>
            MediAI Diagnostics — GLS University Capstone Project 2025-26 |
            Integrated MSc(IT) Programme
          </p>
        </div>
      </div>
    </div>
  )
}