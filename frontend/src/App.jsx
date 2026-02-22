import { useState } from 'react'
import axios from 'axios'
import './App.css'

function App() {
  const [image, setImage] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [symptoms, setSymptoms] = useState('')
  const [vitals, setVitals] = useState({
    pregnancies: '', glucose: '', blood_pressure: '',
    skin_thickness: '', insulin: '', bmi: '',
    diabetes_pedigree: '', age: ''
  })
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleImage = (e) => {
    const file = e.target.files[0]
    if (!file) return
    if (!file.type.startsWith('image/')) {
      setError('Please upload a valid image file!')
      return
    }
    setImage(file)
    setImagePreview(URL.createObjectURL(file))
    setError(null)
  }

  const handleVitals = (e) => {
    setVitals({ ...vitals, [e.target.name]: e.target.value })
  }

  const validateForm = () => {
    if (!image) { setError('Please upload an X-ray image!'); return false }
    if (!symptoms.trim()) { setError('Please enter patient symptoms!'); return false }
    if (!vitals.glucose || !vitals.age || !vitals.bmi) {
      setError('Please fill in at least Glucose, BMI and Age!'); return false
    }
    return true
  }

  const handleSubmit = async () => {
    setError(null)
    if (!validateForm()) return
    setLoading(true)
    setResult(null)

    const formData = new FormData()
    formData.append('image', image)
    formData.append('symptoms', symptoms)
    Object.keys(vitals).forEach(key => formData.append(key, vitals[key] || 0))

    try {
      const res = await axios.post('http://127.0.0.1:8000/predict', formData, {
        timeout: 30000
      })
      setResult(res.data)
    } catch (err) {
      if (err.code === 'ECONNABORTED') {
        setError('Request timed out. Please try again!')
      } else if (err.response) {
        setError(`Server error: ${err.response.status}`)
      } else {
        setError('Cannot connect to backend. Make sure the server is running!')
      }
    }
    setLoading(false)
  }

  const handleReset = () => {
    setImage(null)
    setImagePreview(null)
    setSymptoms('')
    setVitals({
      pregnancies: '', glucose: '', blood_pressure: '',
      skin_thickness: '', insulin: '', bmi: '',
      diabetes_pedigree: '', age: ''
    })
    setResult(null)
    setError(null)
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

  return (
    <div style={{ maxWidth: '900px', margin: '0 auto', padding: '20px', fontFamily: 'Arial' }}>
      <h1 style={{ textAlign: 'center', color: '#1e40af' }}>
        🏥 AI Healthcare Diagnosis System
      </h1>
      <p style={{ textAlign: 'center', color: '#64748b' }}>
        Upload X-ray, enter vitals & symptoms for AI-powered diagnosis
      </p>

      {/* Error Box */}
      {error && (
        <div style={{
          background: '#fee2e2', border: '1px solid #ef4444', padding: '12px',
          borderRadius: '8px', marginBottom: '20px', color: '#dc2626'
        }}>
          ⚠️ {error}
        </div>
      )}

      {/* Image Upload */}
      <div style={{ background: '#f1f5f9', padding: '20px', borderRadius: '10px', marginBottom: '20px' }}>
        <h2>📷 Upload X-Ray Image</h2>
        <input type="file" accept="image/*" onChange={handleImage} />
        {imagePreview && (
          <img src={imagePreview} alt="X-ray preview"
            style={{ width: '200px', marginTop: '10px', borderRadius: '8px', display: 'block' }} />
        )}
      </div>

      {/* Vitals */}
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
              <input
                type="number" name={key}
                value={vitals[key]} onChange={handleVitals}
                placeholder={`Enter ${label}`}
                style={{
                  width: '100%', padding: '8px', marginTop: '4px',
                  borderRadius: '6px', border: '1px solid #cbd5e1', boxSizing: 'border-box'
                }}
              />
            </div>
          ))}
        </div>
        <p style={{ color: '#94a3b8', fontSize: '12px' }}>* Required fields</p>
      </div>

      {/* Symptoms */}
      <div style={{ background: '#f1f5f9', padding: '20px', borderRadius: '10px', marginBottom: '20px' }}>
        <h2>📝 Symptoms *</h2>
        <textarea
          value={symptoms} onChange={(e) => setSymptoms(e.target.value)}
          placeholder="Describe patient symptoms e.g. fever, cough, difficulty breathing..."
          style={{
            width: '100%', height: '100px', padding: '10px',
            borderRadius: '6px', border: '1px solid #cbd5e1', boxSizing: 'border-box'
          }}
        />
      </div>

      {/* Buttons */}
      <div style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
        <button onClick={handleSubmit} disabled={loading}
          style={{
            flex: 1, padding: '15px', background: loading ? '#93c5fd' : '#1e40af',
            color: 'white', border: 'none', borderRadius: '10px',
            fontSize: '18px', cursor: loading ? 'not-allowed' : 'pointer'
          }}>
          {loading ? '🔄 Analyzing... Please wait' : '🔍 Analyze Patient'}
        </button>
        <button onClick={handleReset}
          style={{
            padding: '15px 25px', background: '#64748b',
            color: 'white', border: 'none', borderRadius: '10px',
            fontSize: '16px', cursor: 'pointer'
          }}>
          🔁 Reset
        </button>
      </div>

      {/* Results */}
      {result && (
        <div style={{
          background: '#f0fdf4', padding: '20px', borderRadius: '10px',
          border: '2px solid #22c55e'
        }}>
          <h2>📊 Diagnosis Results</h2>
          <h3 style={{ color: '#1e40af', fontSize: '24px' }}>
            Diagnosis: {result.diagnosis}
          </h3>

          {/* Risk Score */}
          <div style={{ marginBottom: '20px' }}>
            <p style={{ fontWeight: 'bold', fontSize: '18px' }}>
              Clinical Risk Score: {result.risk_score}/100 — {getRiskLabel(result.risk_score)}
            </p>
            <div style={{ background: '#e2e8f0', borderRadius: '10px', height: '25px' }}>
              <div style={{
                width: `${result.risk_score}%`, height: '100%',
                background: getRiskColor(result.risk_score),
                borderRadius: '10px', transition: 'width 0.8s ease'
              }} />
            </div>
          </div>

          {/* Probabilities */}
          <h3>Disease Probabilities:</h3>
          {Object.entries(result.probabilities).map(([disease, prob]) => (
            <div key={disease} style={{ marginBottom: '12px' }}>
              <p style={{ margin: '0 0 4px', fontWeight: 'bold' }}>
                {disease}: {prob}%
              </p>
              <div style={{ background: '#e2e8f0', borderRadius: '10px', height: '18px' }}>
                <div style={{
                  width: `${prob}%`, height: '100%',
                  background: '#1e40af', borderRadius: '10px',
                  transition: 'width 0.8s ease'
                }} />
              </div>
            </div>
          ))}

          <p style={{ color: '#64748b', fontSize: '12px', marginTop: '15px' }}>
            ⚠️ This is an AI-assisted diagnosis. Always consult a qualified medical professional.
          </p>
        </div>
      )}
    </div>
  )
}

export default App