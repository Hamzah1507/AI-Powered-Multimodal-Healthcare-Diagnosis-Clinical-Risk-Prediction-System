import { useState } from 'react'
import axios from 'axios'

function App() {
  const [activeModule, setActiveModule] = useState('xray')
  const [image, setImage] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [patientName, setPatientName] = useState('')
  const [patientAge, setPatientAge] = useState('')
  const [patientGender, setPatientGender] = useState('Male')
  const [vitals, setVitals] = useState({
    pregnancies: '', glucose: '', blood_pressure: '',
    skin_thickness: '', insulin: '', bmi: '',
    diabetes_pedigree: '', age: ''
  })
  const [xrayResult, setXrayResult] = useState(null)
  const [brainResult, setBrainResult] = useState(null)
  const [vitalsResult, setVitalsResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleImage = (e) => {
    const file = e.target.files[0]
    if (!file) return
    setImage(file)
    setImagePreview(URL.createObjectURL(file))
    setError(null)
  }

  const handleReset = () => {
    setImage(null); setImagePreview(null)
    setXrayResult(null); setBrainResult(null)
    setVitalsResult(null); setError(null)
    setPatientName(''); setPatientAge(''); setPatientGender('Male')
    setVitals({ pregnancies: '', glucose: '', blood_pressure: '',
      skin_thickness: '', insulin: '', bmi: '', diabetes_pedigree: '', age: '' })
  }

  const handleSubmit = async () => {
    setError(null)
    if (!image) { setError('Please upload an image!'); return }
    if (activeModule === 'xray' && (!vitals.glucose || !vitals.age || !vitals.bmi)) {
      setError('Please fill in Glucose, BMI and Age!'); return
    }
    setLoading(true)
    setXrayResult(null); setBrainResult(null); setVitalsResult(null)
    try {
      const imageForm = new FormData()
      imageForm.append('image', image)
      if (activeModule === 'xray') {
        const vitalsForm = new FormData()
        Object.keys(vitals).forEach(key => vitalsForm.append(key, vitals[key] || 0))
        const [xrayRes, vitalsRes] = await Promise.all([
          axios.post('http://127.0.0.1:8000/predict-xray', imageForm),
          axios.post('http://127.0.0.1:8000/predict-vitals', vitalsForm)
        ])
        setXrayResult(xrayRes.data)
        setVitalsResult(vitalsRes.data)
      } else {
        const brainRes = await axios.post('http://127.0.0.1:8000/predict-brain', imageForm)
        if (brainRes.data.status === 'error') {
          setError(brainRes.data.message)
        } else {
          setBrainResult(brainRes.data)
        }
      }
    } catch (err) {
      setError('Error connecting to backend!')
    }
    setLoading(false)
  }

  const getRiskColor = (score) => {
    if (score >= 70) return '#ef4444'
    if (score >= 40) return '#f97316'
    return '#22c55e'
  }

  const getRiskLabel = (score) => {
    if (score >= 70) return '🔴 High Risk'
    if (score >= 40) return '🟡 Medium Risk'
    return '🟢 Low Risk'
  }

  const card = {
    background: 'rgba(255,255,255,0.05)',
    border: '1px solid rgba(255,255,255,0.1)',
    borderRadius: '16px',
    padding: '24px',
    marginBottom: '20px',
    backdropFilter: 'blur(10px)'
  }

  const input = {
    width: '100%',
    padding: '10px 14px',
    borderRadius: '8px',
    border: '1px solid rgba(255,255,255,0.15)',
    background: 'rgba(255,255,255,0.07)',
    color: 'white',
    fontSize: '14px',
    marginTop: '6px',
    boxSizing: 'border-box',
    outline: 'none'
  }

  const label = {
    color: '#94a3b8',
    fontSize: '13px',
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: '0.5px'
  }

  const ResultCard = ({ title, result, color }) => (
    <div style={{ ...card, border: `1px solid ${color}40`, flex: 1 }}>
      <h3 style={{ color, marginBottom: '16px', fontSize: '16px' }}>{title}</h3>
      <div style={{ background: `${color}15`, borderRadius: '10px',
        padding: '12px', marginBottom: '16px' }}>
        <p style={{ color: 'white', fontSize: '22px', fontWeight: 'bold', margin: 0 }}>
          {result.diagnosis}
        </p>
        <p style={{ color: getRiskColor(result.risk_score), margin: '4px 0 0', fontWeight: '600' }}>
          {getRiskLabel(result.risk_score)} — {result.risk_score}/100
        </p>
      </div>
      <div style={{ background: 'rgba(255,255,255,0.1)', borderRadius: '8px',
        height: '8px', marginBottom: '20px' }}>
        <div style={{ width: `${result.risk_score}%`, height: '100%',
          background: getRiskColor(result.risk_score), borderRadius: '8px',
          transition: 'width 1s ease' }} />
      </div>
      {Object.entries(result.probabilities).map(([disease, prob]) => (
        <div key={disease} style={{ marginBottom: '12px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
            <span style={{ color: '#cbd5e1', fontSize: '13px' }}>{disease}</span>
            <span style={{ color: 'white', fontWeight: 'bold', fontSize: '13px' }}>{prob}%</span>
          </div>
          <div style={{ background: 'rgba(255,255,255,0.1)', borderRadius: '4px', height: '6px' }}>
            <div style={{ width: `${prob}%`, height: '100%',
              background: color, borderRadius: '4px', transition: 'width 1s ease' }} />
          </div>
        </div>
      ))}
    </div>
  )

  return (
    <div style={{ color: 'white', fontFamily: "'Segoe UI', Arial, sans-serif" }}>

      {/* Header */}
      <div style={{ textAlign: 'center', marginBottom: '32px', paddingTop: '20px' }}>
        <div style={{ fontSize: '48px', marginBottom: '8px' }}>🏥</div>
        <h1 style={{ fontSize: '32px', fontWeight: '800', background:
          'linear-gradient(135deg, #60a5fa, #a78bfa)', WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent', margin: 0 }}>
          AI Healthcare Diagnosis System
        </h1>
        <p style={{ color: '#64748b', marginTop: '8px', fontSize: '15px' }}>
          Multimodal AI-powered clinical decision support
        </p>
      </div>

      {/* Module Tabs */}
      <div style={{ display: 'flex', gap: '12px', marginBottom: '24px' }}>
        <button onClick={() => { setActiveModule('xray'); handleReset() }}
          style={{ flex: 1, padding: '16px', border: 'none', borderRadius: '12px',
            fontSize: '15px', cursor: 'pointer', fontWeight: '700', transition: 'all 0.3s',
            background: activeModule === 'xray'
              ? 'linear-gradient(135deg, #1d4ed8, #3b82f6)' : 'rgba(255,255,255,0.05)',
            color: 'white', boxShadow: activeModule === 'xray'
              ? '0 4px 20px rgba(59,130,246,0.4)' : 'none' }}>
          🫁 Chest X-Ray + Diabetes
        </button>
        <button onClick={() => { setActiveModule('brain'); handleReset() }}
          style={{ flex: 1, padding: '16px', border: 'none', borderRadius: '12px',
            fontSize: '15px', cursor: 'pointer', fontWeight: '700', transition: 'all 0.3s',
            background: activeModule === 'brain'
              ? 'linear-gradient(135deg, #6d28d9, #8b5cf6)' : 'rgba(255,255,255,0.05)',
            color: 'white', boxShadow: activeModule === 'brain'
              ? '0 4px 20px rgba(139,92,246,0.4)' : 'none' }}>
          🧠 Brain MRI Tumor Detection
        </button>
      </div>

      {/* Error */}
      {error && (
        <div style={{ background: 'rgba(239,68,68,0.15)', border: '1px solid #ef4444',
          padding: '14px', borderRadius: '10px', marginBottom: '20px',
          color: '#fca5a5', fontSize: '14px' }}>
          {error}
        </div>
      )}

      {/* Patient Info */}
      <div style={card}>
        <h2 style={{ color: '#60a5fa', marginBottom: '16px', fontSize: '16px' }}>
          👤 Patient Information
        </h2>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '16px' }}>
          <div>
            <label style={label}>Patient Name</label>
            <input style={input} placeholder="Enter name"
              value={patientName} onChange={e => setPatientName(e.target.value)} />
          </div>
          <div>
            <label style={label}>Age</label>
            <input style={input} type="number" placeholder="Enter age"
              value={patientAge} onChange={e => setPatientAge(e.target.value)} />
          </div>
          <div>
            <label style={label}>Gender</label>
            <select style={input} value={patientGender}
              onChange={e => setPatientGender(e.target.value)}>
              <option>Male</option>
              <option>Female</option>
              <option>Other</option>
            </select>
          </div>
        </div>
      </div>

      {/* Image Upload */}
      <div style={card}>
        <h2 style={{ color: activeModule === 'brain' ? '#a78bfa' : '#60a5fa',
          marginBottom: '16px', fontSize: '16px' }}>
          {activeModule === 'xray' ? '📷 Chest X-Ray Image' : '🧠 Brain MRI Scan'}
        </h2>
        <label style={{ display: 'block', border: '2px dashed rgba(255,255,255,0.2)',
          borderRadius: '12px', padding: '30px', textAlign: 'center', cursor: 'pointer',
          transition: 'all 0.3s' }}>
          <input type="file" accept="image/*" onChange={handleImage}
            style={{ display: 'none' }} />
          {imagePreview ? (
            <img src={imagePreview} alt="preview"
              style={{ maxWidth: '250px', maxHeight: '250px',
                borderRadius: '10px', objectFit: 'contain' }} />
          ) : (
            <div>
              <div style={{ fontSize: '40px', marginBottom: '10px' }}>📁</div>
              <p style={{ color: '#64748b' }}>Click to upload image</p>
              <p style={{ color: '#475569', fontSize: '12px', marginTop: '4px' }}>
                PNG, JPG, JPEG supported
              </p>
            </div>
          )}
        </label>
        {activeModule === 'brain' && (
          <p style={{ color: '#a78bfa', fontSize: '12px', marginTop: '10px' }}>
            ⚠️ Only Brain MRI scans accepted. Chest X-rays and colorful images will be rejected.
          </p>
        )}
      </div>

      {/* Vitals */}
      {activeModule === 'xray' && (
        <div style={card}>
          <h2 style={{ color: '#60a5fa', marginBottom: '16px', fontSize: '16px' }}>
            💉 Patient Vitals
          </h2>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
            {[
              ['pregnancies', 'Pregnancies'],
              ['glucose', 'Glucose *'],
              ['blood_pressure', 'Blood Pressure'],
              ['skin_thickness', 'Skin Thickness'],
              ['insulin', 'Insulin'],
              ['bmi', 'BMI *'],
              ['diabetes_pedigree', 'Diabetes Pedigree'],
              ['age', 'Age *']
            ].map(([key, lbl]) => (
              <div key={key}>
                <label style={label}>{lbl}</label>
                <input type="number" style={input}
                  placeholder={`Enter ${lbl.replace(' *','')}`}
                  name={key} value={vitals[key]}
                  onChange={e => setVitals({...vitals, [key]: e.target.value})} />
              </div>
            ))}
          </div>
          <p style={{ color: '#475569', fontSize: '11px', marginTop: '12px' }}>
            * Required fields
          </p>
        </div>
      )}

      {/* Analyze Button */}
      <div style={{ display: 'flex', gap: '12px', marginBottom: '24px' }}>
        <button onClick={handleSubmit} disabled={loading} style={{
          flex: 1, padding: '16px', border: 'none', borderRadius: '12px',
          fontSize: '17px', fontWeight: '700', cursor: loading ? 'not-allowed' : 'pointer',
          background: loading ? 'rgba(255,255,255,0.1)' :
            activeModule === 'brain'
              ? 'linear-gradient(135deg, #6d28d9, #8b5cf6)'
              : 'linear-gradient(135deg, #1d4ed8, #3b82f6)',
          color: 'white',
          boxShadow: loading ? 'none' :
            activeModule === 'brain'
              ? '0 4px 20px rgba(139,92,246,0.4)'
              : '0 4px 20px rgba(59,130,246,0.4)'
        }}>
          {loading ? '🔄 Analyzing Patient Data...' : '🔍 Analyze Patient'}
        </button>
        <button onClick={handleReset} style={{
          padding: '16px 24px', border: '1px solid rgba(255,255,255,0.15)',
          borderRadius: '12px', fontSize: '15px', cursor: 'pointer',
          background: 'rgba(255,255,255,0.05)', color: '#94a3b8', fontWeight: '600'
        }}>
          🔁 Reset
        </button>
      </div>

      {/* X-Ray Results */}
      {xrayResult && vitalsResult && (
        <div>
          {patientName && (
            <div style={{ textAlign: 'center', marginBottom: '16px' }}>
              <p style={{ color: '#64748b', fontSize: '14px' }}>
                Patient: <span style={{ color: 'white', fontWeight: '700' }}>
                  {patientName}</span> | Age: {patientAge} | Gender: {patientGender}
              </p>
            </div>
          )}
          <h2 style={{ textAlign: 'center', marginBottom: '16px', color: '#60a5fa' }}>
            📊 Diagnosis Results
          </h2>
          <div style={{ display: 'flex', gap: '16px' }}>
            <ResultCard title="🫁 Chest X-Ray Analysis"
              result={xrayResult} color="#3b82f6" />
            <ResultCard title="🩸 Diabetes Risk Analysis"
              result={vitalsResult} color="#10b981" />
          </div>
          <p style={{ color: '#475569', fontSize: '12px', textAlign: 'center', marginTop: '16px' }}>
            ⚠️ AI-assisted diagnosis only. Always consult a qualified medical professional.
          </p>
        </div>
      )}

      {/* Brain Results */}
      {brainResult && (
        <div>
          {patientName && (
            <div style={{ textAlign: 'center', marginBottom: '16px' }}>
              <p style={{ color: '#64748b', fontSize: '14px' }}>
                Patient: <span style={{ color: 'white', fontWeight: '700' }}>
                  {patientName}</span> | Age: {patientAge} | Gender: {patientGender}
              </p>
            </div>
          )}
          <h2 style={{ textAlign: 'center', marginBottom: '16px', color: '#a78bfa' }}>
            📊 Brain MRI Results
          </h2>
          <ResultCard title="🧠 Brain Tumor Analysis"
            result={brainResult} color="#8b5cf6" />
          <p style={{ color: '#475569', fontSize: '12px', textAlign: 'center', marginTop: '16px' }}>
            ⚠️ AI-assisted diagnosis only. Always consult a qualified medical professional.
          </p>
        </div>
      )}

      {/* Footer */}
      <div style={{ textAlign: 'center', marginTop: '40px', paddingBottom: '20px',
        borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '20px' }}>
        <p style={{ color: '#334155', fontSize: '12px' }}>
          AI Healthcare Diagnosis System — Capstone Project 2026
        </p>
      </div>
    </div>
  )
}

export default App