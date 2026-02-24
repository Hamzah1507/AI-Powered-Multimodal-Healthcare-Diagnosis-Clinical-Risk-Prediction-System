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
const [xrayHeatmap, setXrayHeatmap] = useState(null)
const [brainHeatmap, setBrainHeatmap] = useState(null)

const [loading, setLoading] = useState(false)
const [error, setError] = useState(null)

// ───────────────── SCREEN ROUTING ─────────────────
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

// ───────────────── HELPERS ─────────────────
const reset = () => {
setImage(null)
setPreview(null)
setError(null)
setXrayResult(null)
setVitalsResult(null)
setBrainResult(null)
setXrayHeatmap(null)
setBrainHeatmap(null)
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

// ───────────────── ANALYZE ─────────────────
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

// ───────────────── UI ─────────────────
return ( <div className="app-bg">

```
  {/* NAVBAR */}
  <nav className="navbar">
    <div className="nav-left">
      <div className="logo-box">🏥</div>
      <div>
        <h1 className="brand-title">MediAI Diagnostics</h1>
        <p className="brand-sub">AI Clinical Decision Support</p>
      </div>
    </div>

    <div className="nav-right">
      <button
        className={`module-btn ${module === 'xray' ? 'active-blue' : ''}`}
        onClick={() => switchModule('xray')}
      >
        🫁 Chest X-Ray
      </button>

      <button
        className={`module-btn ${module === 'brain' ? 'active-purple' : ''}`}
        onClick={() => switchModule('brain')}
      >
        🧠 Brain MRI
      </button>

      {user && (
        <div className="user-badge">
          👤 {user.username}
        </div>
      )}

      <button
        className="logout-btn"
        onClick={() => {
          reset()
          setScreen('welcome')
          setUser(null)
        }}
      >
        Logout
      </button>
    </div>
  </nav>

  {/* HEADER */}
  <header className={`hero ${module}`}>
    <div className="container">
      <h2 className="hero-title">
        {module === 'xray'
          ? 'Chest X-Ray + Diabetes Analysis'
          : 'Brain MRI Tumor Detection'}
      </h2>

      <p className="hero-sub">
        AI-powered clinical decision support system
      </p>
    </div>
  </header>

  {/* MAIN */}
  <main className="container main-grid">

    {/* LEFT */}
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

      <div className="grid-2">
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

    {/* RIGHT */}
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
            🖼️ Click to upload
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

      {error && <div className="error-box">⚠️ {error}</div>}
    </div>

  </main>
</div>
```

)
}
