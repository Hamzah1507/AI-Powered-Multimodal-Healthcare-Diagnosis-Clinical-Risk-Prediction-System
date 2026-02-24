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

const [patient, setPatient] = useState({
name: '',
age: '',
gender: 'Male',
id: ''
})

const [vitals, setVitals] = useState({
pregnancies: '',
glucose: '',
blood_pressure: '',
skin_thickness: '',
insulin: '',
bmi: '',
diabetes_pedigree: '',
age: ''
})

const [xrayResult, setXrayResult] = useState(null)
const [vitalsResult, setVitalsResult] = useState(null)
const [brainResult, setBrainResult] = useState(null)

const [loading, setLoading] = useState(false)
const [error, setError] = useState(null)

// ───────────── SCREEN ROUTING ─────────────
if (screen === 'welcome') {
return (
<Welcome
onLogin={() => { setAuthMode('login'); setScreen('auth') }}
onRegister={() => { setAuthMode('register'); setScreen('auth') }}
/>
)
}

if (screen === 'auth') {
return (
<Auth
mode={authMode}
onSuccess={(u) => { setUser(u); setScreen('dashboard') }}
onBack={() => setScreen('welcome')}
/>
)
}

// ───────────── HELPERS ─────────────
const reset = () => {
setImage(null)
setPreview(null)
setError(null)
setXrayResult(null)
setVitalsResult(null)
setBrainResult(null)
}

const switchModule = (m) => {
setModule(m)
reset()
}

const handleImage = (e) => {
const file = e.target.files[0]
if (!file) return
setImage(file)
setPreview(URL.createObjectURL(file))
}

// ───────────── ANALYZE ─────────────
const analyze = async () => {
setError(null)

```
if (!image) {
  setError('Please upload an image first!')
  return
}

setLoading(true)

try {
  const imgForm = new FormData()
  imgForm.append('image', image)

  if (module === 'xray') {
    const vForm = new FormData()
    Object.keys(vitals).forEach(k =>
      vForm.append(k, vitals[k] || 0)
    )

    const [xr, vr] = await Promise.all([
      axios.post(`${API}/predict-xray`, imgForm),
      axios.post(`${API}/predict-vitals`, vForm)
    ])

    setXrayResult(xr.data)
    setVitalsResult(vr.data)

  } else {
    const br = await axios.post(`${API}/predict-brain`, imgForm)
    setBrainResult(br.data)
  }

} catch {
  setError('Cannot connect to backend. Make sure server is running!')
}

setLoading(false)
```

}

// ───────────── UI ─────────────
return ( <div className="dashboard-layout">

```
  {/* SIDEBAR */}
  <aside className="sidebar">

    <div>
      <div className="sidebar-top">
        <div className="logo">🏥</div>
        <div>
          <h2 className="logo-title">MediAI</h2>
          <p className="logo-sub">Diagnostics</p>
        </div>
      </div>

      <div className="sidebar-menu">
        <button
          className={`side-btn ${module === 'xray' ? 'active' : ''}`}
          onClick={() => switchModule('xray')}
        >
          🫁 Chest X-Ray
        </button>

        <button
          className={`side-btn ${module === 'brain' ? 'active' : ''}`}
          onClick={() => switchModule('brain')}
        >
          🧠 Brain MRI
        </button>
      </div>
    </div>

    <div className="sidebar-bottom">
      {user && (
        <div className="user-pill">
          👤 {user.username}
        </div>
      )}

      <button
        className="logout-side"
        onClick={() => {
          reset()
          setScreen('welcome')
          setUser(null)
        }}
      >
        🚪 Logout
      </button>
    </div>

  </aside>

  {/* MAIN AREA */}
  <main className="main-area">

    {/* HERO */}
    <div className={`page-hero ${module}`}>
      <h1>
        {module === 'xray'
          ? 'Chest X-Ray + Diabetes Analysis'
          : 'Brain MRI Tumor Detection'}
      </h1>
      <p>AI-powered clinical decision support system</p>
    </div>

    {/* CONTENT */}
    <div className="content-wrap">

      {error && <div className="error-box">⚠️ {error}</div>}

      <div className="grid-2">

        {/* PATIENT CARD */}
        <div className="card">
          <h3 className="card-title">Patient Information</h3>

          <input
            className="input"
            placeholder="Full name"
            value={patient.name}
            onChange={e =>
              setPatient({ ...patient, name: e.target.value })
            }
          />

          <div className="grid-2-inner">
            <input
              className="input"
              placeholder="Patient ID"
              value={patient.id}
              onChange={e =>
                setPatient({ ...patient, id: e.target.value })
              }
            />

            <input
              className="input"
              type="number"
              placeholder="Age"
              value={patient.age}
              onChange={e =>
                setPatient({ ...patient, age: e.target.value })
              }
            />
          </div>

          <select
            className="input"
            value={patient.gender}
            onChange={e =>
              setPatient({ ...patient, gender: e.target.value })
            }
          >
            <option>Male</option>
            <option>Female</option>
            <option>Other</option>
          </select>
        </div>

        {/* UPLOAD CARD */}
        <div className="card">
          <h3 className="card-title">
            {module === 'xray' ? 'Upload X-Ray' : 'Upload Brain MRI'}
          </h3>

          <label className="upload-box">
            <input
              type="file"
              accept="image/*"
              onChange={handleImage}
              hidden
            />

            {preview ? (
              <img src={preview} className="preview-img" />
            ) : (
              <div className="upload-placeholder">
                🖼️ Click to upload image
              </div>
            )}
          </label>

          <button
            className="analyze-btn"
            disabled={loading}
            onClick={analyze}
          >
            {loading ? 'Analyzing…' : 'Run AI Diagnosis'}
          </button>
        </div>

      </div>

      {/* RESULTS */}
      {(xrayResult || brainResult) && (
        <div className="card" style={{ marginTop: 24 }}>
          <h3 className="card-title">AI Diagnosis Result</h3>

          {xrayResult && (
            <p><strong>X-Ray:</strong> {xrayResult.diagnosis}</p>
          )}

          {vitalsResult && (
            <p><strong>Diabetes:</strong> {vitalsResult.diagnosis}</p>
          )}

          {brainResult && (
            <p><strong>Brain MRI:</strong> {brainResult.diagnosis}</p>
          )}
        </div>
      )}

    </div>
  </main>
</div>
```

)
}
