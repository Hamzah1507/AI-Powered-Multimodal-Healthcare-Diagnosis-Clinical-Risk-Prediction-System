import { useState } from 'react'
import axios from 'axios'

const API = 'http://127.0.0.1:8000'

export default function Auth({ mode, onSuccess, onBack }) {
  const [form, setForm] = useState({ username: '', email: '', password: '' })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [isLogin, setIsLogin] = useState(mode === 'login')

  const handle = async () => {
    setError(null)
    if (!form.email || !form.password) { setError('Please fill all required fields!'); return }
    if (!isLogin && !form.username) { setError('Please enter your name!'); return }
    setLoading(true)
    try {
      const endpoint = isLogin ? '/auth/login' : '/auth/register'
      const payload = isLogin
        ? { email: form.email, password: form.password }
        : { username: form.username, email: form.email, password: form.password }
      const res = await axios.post(`${API}${endpoint}`, payload)
      if (!isLogin) {
        setIsLogin(true)
        setError(null)
        setForm({ ...form, password: '' })
        alert('✅ Account created! Please login now.')
      } else {
        onSuccess({ username: res.data.username, email: res.data.email })
      }
    } catch (e) {
      setError(e.response?.data?.detail || 'Something went wrong!')
    }
    setLoading(false)
  }

  const inp = {
    width: '100%', padding: '12px 16px', borderRadius: '10px',
    border: '1.5px solid rgba(255,255,255,0.15)',
    background: 'rgba(255,255,255,0.07)',
    color: 'white', fontSize: '14px', boxSizing: 'border-box',
    outline: 'none', marginTop: '6px'
  }
  const lbl = {
    fontSize: '12px', fontWeight: '600',
    color: 'rgba(255,255,255,0.6)',
    textTransform: 'uppercase', letterSpacing: '0.5px'
  }

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #0f172a 100%)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontFamily: '-apple-system, BlinkMacSystemFont, sans-serif'
    }}>
      <div style={{
        background: 'rgba(255,255,255,0.05)',
        backdropFilter: 'blur(20px)',
        border: '1px solid rgba(255,255,255,0.1)',
        borderRadius: '24px', padding: '48px',
        width: '100%', maxWidth: '440px',
        boxShadow: '0 25px 50px rgba(0,0,0,0.5)'
      }}>

        {/* Header */}
        <div style={{ textAlign: 'center', marginBottom: '32px' }}>
          <div style={{
            background: 'linear-gradient(135deg, #2563eb, #0ea5e9)',
            borderRadius: '16px', padding: '12px',
            fontSize: '28px', display: 'inline-block', marginBottom: '16px'
          }}>🏥</div>
          <h2 style={{ color: 'white', fontSize: '24px', fontWeight: '800', marginBottom: '6px' }}>
            {isLogin ? 'Welcome Back!' : 'Create Account'}
          </h2>
          <p style={{ color: 'rgba(255,255,255,0.5)', fontSize: '14px' }}>
            {isLogin ? 'Login to MediAI Diagnostics' : 'Join MediAI Diagnostics'}
          </p>
        </div>

        {/* Error */}
        {error && (
          <div style={{
            background: 'rgba(220,38,38,0.2)', border: '1px solid rgba(220,38,38,0.4)',
            borderRadius: '10px', padding: '12px 16px', marginBottom: '20px',
            color: '#fca5a5', fontSize: '14px'
          }}>⚠️ {error}</div>
        )}

        {/* Form */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {!isLogin && (
            <div>
              <label style={lbl}>Full Name</label>
              <input style={inp} placeholder="Dr. John Smith"
                value={form.username}
                onChange={e => setForm({ ...form, username: e.target.value })} />
            </div>
          )}
          <div>
            <label style={lbl}>Email Address</label>
            <input style={inp} type="email" placeholder="doctor@hospital.com"
              value={form.email}
              onChange={e => setForm({ ...form, email: e.target.value })} />
          </div>
          <div>
            <label style={lbl}>Password</label>
            <input style={inp} type="password" placeholder="••••••••"
              value={form.password}
              onChange={e => setForm({ ...form, password: e.target.value })}
              onKeyDown={e => e.key === 'Enter' && handle()} />
          </div>
        </div>

        {/* Submit Button */}
        <button onClick={handle} disabled={loading} style={{
          width: '100%', padding: '14px', marginTop: '24px',
          background: loading ? 'rgba(255,255,255,0.1)' : 'linear-gradient(135deg, #2563eb, #0ea5e9)',
          border: 'none', borderRadius: '12px',
          color: loading ? 'rgba(255,255,255,0.4)' : 'white',
          fontSize: '16px', fontWeight: '700',
          cursor: loading ? 'not-allowed' : 'pointer',
          boxShadow: loading ? 'none' : '0 4px 20px rgba(37,99,235,0.4)'
        }}>
          {loading ? '⏳ Please wait...' : isLogin ? '🔐 Login' : '✨ Create Account'}
        </button>

        {/* Switch Mode */}
        <p style={{ textAlign: 'center', marginTop: '20px',
          color: 'rgba(255,255,255,0.5)', fontSize: '14px' }}>
          {isLogin ? "Don't have an account? " : "Already have an account? "}
          <span onClick={() => { setIsLogin(!isLogin); setError(null) }} style={{
            color: '#60a5fa', fontWeight: '700', cursor: 'pointer'
          }}>
            {isLogin ? 'Register' : 'Login'}
          </span>
        </p>

        {/* Back Button */}
        <p style={{ textAlign: 'center', marginTop: '12px' }}>
          <span onClick={onBack} style={{
            color: 'rgba(255,255,255,0.4)', fontSize: '13px', cursor: 'pointer'
          }}>
            ← Back to Welcome
          </span>
        </p>
      </div>
    </div>
  )
}