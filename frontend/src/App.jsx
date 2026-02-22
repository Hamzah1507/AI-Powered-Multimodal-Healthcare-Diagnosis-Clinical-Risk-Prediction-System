import { useState } from 'react'
import axios from 'axios'
import './App.css'

function App() {
  const [activeModule, setActiveModule] = useState('xray')
  const [image, setImage] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
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

  const handleVitals = (e) => {
    setVitals({ ...vitals, [e.target.name]: e.target.value })
  }

  const handleReset = () => {
    setImage(null); setImagePreview(null)
    setXrayResult(null); setBrainResult(null)
    setVitalsResult(null); setError(null)
    setVitals({
      pregnancies: '', glucose: '', blood_pressure: '',
      skin_thickness: '', insulin: '', bmi: '', diabetes_pedigree: '', age: ''
    })
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
        setBrainResult(brainRes.data)
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

  const ResultCard = ({ title, result }) => (
    <div style={{
      background: '#f0fdf4', padding: '20px', borderRadius: '10px',
      border: '2px solid #22c55e', flex: 1
    }}>
      <h3 style={{ color: '#1e40af', marginBottom: '10px' }}>{title}</h3>
      <h4 style={{ fontSize: '20px', marginBottom: '10px' }}>
        Diagnosis: {result.diagnosis}
      </h4>
      <p style={{ fontWeight: 'bold' }}>
        Risk Score: {result.risk_score}/100 — {getRiskLabel(result.risk_score)}
      </p>
      <div style={{
        background: '#e2e8f0', borderRadius: '10px',
        height: '20px', margin: '8px 0 16px'
      }}>
        <div style={{
          width: `${result.risk_score}%`, height: '100%',
          background: getRiskColor(result.risk_score), borderRadius: '10px'
        }} />
      </div>
      {Object.entries(result.probabilities).map(([disease, prob]) => (
        <div key={disease} style={{ marginBottom: '10px' }}>
          <p style={{ margin: '0 0 4px', fontWeight: 'bold' }}>{disease}: {prob}%</p>
          <div style={{ background: '#e2e8f0', borderRadius: '10px', height: '14px' }}>
            <div style={{
              width: `${prob}%`, height: '100%',
              background: '#1e40af', borderRadius: '10px'
            }} />
          </div>
        </div>
      ))}
    </div>
  )

  return (
    <div style={{ maxWidth: '950px', margin: '0 auto', padding: '20px', fontFamily: 'Arial' }}>
      <h1 style={{ textAlign: 'center', color: '#1e40af' }}>
        🏥 AI Healthcare Diagnosis System
      </h1>
      <p style={{ textAlign: 'center', color: '#64748b' }}>
        Multimodal AI-powered medical diagnosis
      </p>

      {/* Module Selector */}
      <div style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
        <button onClick={() => { setActiveModule('xray'); handleReset() }}
          style={{
            flex: 1, padding: '15px', border: 'none', borderRadius: '10px',
            fontSize: '16px', cursor: 'pointer', fontWeight: 'bold',
            background: activeModule === 'xray' ? '#1e40af' : '#e2e8f0',
            color: activeModule === 'xray' ? 'white' : '#1e40af'
          }}>
          🫁 Chest X-Ray + Diabetes
        </button>
        <button onClick={() => { setActiveModule('brain'); handleReset() }}
          style={{
            flex: 1, padding: '15px', border: 'none', borderRadius: '10px',
            fontSize: '16px', cursor: 'pointer', fontWeight: 'bold',
            background: activeModule === 'brain' ? '#7c3aed' : '#e2e8f0',
            color: activeModule === 'brain' ? 'white' : '#7c3aed'
          }}>
          🧠 Brain MRI Tumor Detection
        </button>
      </div>

      {error && (
        <div style={{
          background: '#fee2e2', border: '1px solid #ef4444',
          padding: '12px', borderRadius: '8px', marginBottom: '20px', color: '#dc2626'
        }}>
          ⚠️ {error}
        </div>
      )}

      {/* Image Upload */}
      <div style={{ background: '#f1f5f9', padding: '20px', borderRadius: '10px', marginBottom: '20px' }}>
        <h2>{activeModule === 'xray' ? '📷 Upload Chest X-Ray' : '🧠 Upload Brain MRI Scan'}</h2>
        <input type="file" accept="image/*" onChange={handleImage} />
        {imagePreview && (
          <img src={imagePreview} alt="preview"
            style={{ width: '200px', marginTop: '10px', borderRadius: '8px', display: 'block' }} />
        )}
      </div>

      {/* Vitals — only for X-Ray module */}
      {activeModule === 'xray' && (
        <div style={{ background: '#f1f5f9', padding: '20px', borderRadius: '10px', marginBottom: '20px' }}>
          <h2>💉 Patient Vitals</h2>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
            {[
              ['pregnancies', 'Pregnancies'],
              ['glucose', 'Glucose *'],
              ['blood_pressure', 'Blood Pressure'],
              ['skin_thickness', 'Skin Thickness'],
              ['insulin', 'Insulin'],
              ['bmi', 'BMI *'],
              ['diabetes_pedigree', 'Diabetes Pedigree'],
              ['age', 'Age *']
            ].map(([key, label]) => (
              <div key={key}>
                <label style={{ fontWeight: 'bold' }}>{label}</label>
                <input type="number" name={key} value={vitals[key]}
                  onChange={handleVitals} placeholder={`Enter ${label}`}
                  style={{
                    width: '100%', padding: '8px', marginTop: '4px',
                    borderRadius: '6px', border: '1px solid #cbd5e1', boxSizing: 'border-box'
                  }} />
              </div>
            ))}
          </div>
          <p style={{ color: '#94a3b8', fontSize: '12px', marginTop: '8px' }}>* Required fields</p>
        </div>
      )}

      {/* Brain MRI info box */}
      {activeModule === 'brain' && (
        <div style={{
          background: '#f5f3ff', padding: '15px', borderRadius: '10px',
          marginBottom: '20px', border: '1px solid #7c3aed'
        }}>
          <p style={{ color: '#7c3aed', fontWeight: 'bold', margin: 0 }}>
            🧠 This module detects: Glioma, Meningioma, Pituitary Tumor, or No Tumor
          </p>
          <p style={{ color: '#64748b', fontSize: '13px', margin: '5px 0 0' }}>
            Upload a brain MRI scan image for tumor detection
          </p>
        </div>
      )}

      {/* Buttons */}
      <div style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
        <button onClick={handleSubmit} disabled={loading}
          style={{
            flex: 1, padding: '15px',
            background: loading ? '#93c5fd' : activeModule === 'brain' ? '#7c3aed' : '#1e40af',
            color: 'white', border: 'none', borderRadius: '10px',
            fontSize: '18px', cursor: loading ? 'not-allowed' : 'pointer'
          }}>
          {loading ? '🔄 Analyzing...' : '🔍 Analyze Patient'}
        </button>
        <button onClick={handleReset}
          style={{
            padding: '15px 25px', background: '#64748b', color: 'white',
            border: 'none', borderRadius: '10px', fontSize: '16px', cursor: 'pointer'
          }}>
          🔁 Reset
        </button>
      </div>

      {/* X-Ray Results */}
      {xrayResult && vitalsResult && (
        <div>
          <h2 style={{ textAlign: 'center' }}>📊 Diagnosis Results</h2>
          <div style={{ display: 'flex', gap: '20px', marginTop: '10px' }}>
            <ResultCard title="🫁 Chest X-Ray Analysis" result={xrayResult} />
            <ResultCard title="🩸 Diabetes Risk Analysis" result={vitalsResult} />
          </div>
          <p style={{ color: '#64748b', fontSize: '12px', marginTop: '15px', textAlign: 'center' }}>
            ⚠️ AI-assisted diagnosis only. Always consult a qualified medical professional.
          </p>
        </div>
      )}

      {/* Brain Results */}
      {brainResult && (
        <div>
          <h2 style={{ textAlign: 'center' }}>📊 Brain MRI Results</h2>
          <div style={{ marginTop: '10px' }}>
            <ResultCard title="🧠 Brain Tumor Analysis" result={brainResult} />
          </div>
          <p style={{ color: '#64748b', fontSize: '12px', marginTop: '15px', textAlign: 'center' }}>
            ⚠️ AI-assisted diagnosis only. Always consult a qualified medical professional.
          </p>
        </div>
      )}
    </div>
  )
}

export default App