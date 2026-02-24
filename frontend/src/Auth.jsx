import { useState } from 'react'
import axios from 'axios'

const API = 'http://127.0.0.1:8000'

export default function Auth({ mode, onSuccess, onBack }) {
  const [form, setForm] = useState({ username: '', email: '', password: '' })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [isLogin, setIsLogin] = useState(mode === 'login')
  const [showPassword, setShowPassword] = useState(false)

  const handle = async () => {
    setError(null)
    if (!form.email || !form.password) { setError('Please fill all required fields'); return }
    if (!isLogin && !form.username) { setError('Please enter your full name'); return }
    setLoading(true)
    try {
      const endpoint = isLogin ? '/auth/login' : '/auth/register'
      const payload = isLogin
        ? { email: form.email, password: form.password }
        : { username: form.username, email: form.email, password: form.password }
      const res = await axios.post(`${API}${endpoint}`, payload)
      if (!isLogin) {
        setIsLogin(true)
        setForm({ ...form, password: '' })
        setError(null)
        alert('Account created successfully! Please sign in.')
      } else {
        onSuccess({ username: res.data.username, email: res.data.email })
      }
    } catch (e) {
      setError(e.response?.data?.detail || 'Something went wrong. Please try again.')
    }
    setLoading(false)
  }

  const inp = {
    width: '100%', padding: '14px 16px',
    border: '1.5px solid #e2e8f0', borderRadius: '6px',
    background: '#fafafa', color: '#0f172a', fontSize: '14px',
    boxSizing: 'border-box', outline: 'none',
    fontFamily: "'Helvetica Neue', Arial, sans-serif", fontWeight: '400',
    transition: 'border-color 0.2s'
  }

  const lbl = {
    fontSize: '11px', fontWeight: '700', color: '#64748b',
    textTransform: 'uppercase', letterSpacing: '1px', display: 'block', marginBottom: '6px'
  }

  return (
    <div style={{
      height: '100vh', width: '100vw', overflow: 'hidden',
      background: '#ffffff', display: 'flex',
      fontFamily: "'Helvetica Neue', Arial, sans-serif"
    }}>
      {/* Left Panel */}
      <div style={{
        width: '40%', height: '100%',
        background: 'linear-gradient(145deg, #eef2ff 0%, #e0e9ff 60%, #dbeafe 100%)',
        display: 'flex', flexDirection: 'column',
        justifyContent: 'center', padding: '80px',
        position: 'relative', overflow: 'hidden'
      }}>
        <div style={{
          position: 'absolute', inset: 0,
          backgroundImage: 'radial-gradient(circle, #2563eb12 1px, transparent 1px)',
          backgroundSize: '28px 28px'
        }} />
        <div style={{
          position: 'absolute', bottom: '-100px', right: '-80px',
          width: '360px', height: '360px', borderRadius: '50%',
          background: 'radial-gradient(circle, #2563eb14 0%, transparent 70%)'
        }} />

        <div style={{ position: 'relative', zIndex: 1 }}>
          <div style={{ marginBottom: '48px' }}>
            <span style={{ fontSize: '11px', fontWeight: '800', letterSpacing: '5px', color: '#2563eb', textTransform: 'uppercase' }}>MediAI</span>
            <div style={{ width: '32px', height: '2px', background: '#2563eb', marginTop: '8px' }} />
          </div>

          <h2 style={{ fontSize: '40px', fontWeight: '300', color: '#0f172a', letterSpacing: '-1.5px', marginBottom: '6px' }}>
            {isLogin ? 'Good to' : 'Join'}
          </h2>
          <h2 style={{ fontSize: '40px', fontWeight: '800', color: '#2563eb', letterSpacing: '-1.5px', marginBottom: '24px' }}>
            {isLogin ? 'see you.' : 'MediAI.'}
          </h2>

          <p style={{ fontSize: '14px', color: '#64748b', lineHeight: 1.8, fontWeight: '300', maxWidth: '280px' }}>
            {isLogin
              ? 'Access AI-powered diagnostics, patient history, and clinical reports.'
              : 'Create your account to start using AI-powered medical imaging analysis.'}
          </p>

          <button onClick={onBack} style={{
            marginTop: '48px', background: 'none', border: 'none',
            cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px',
            color: '#94a3b8', fontSize: '13px', padding: 0, transition: 'color 0.2s'
          }}
            onMouseEnter={e => e.currentTarget.style.color = '#2563eb'}
            onMouseLeave={e => e.currentTarget.style.color = '#94a3b8'}
          >
            ← Back to home
          </button>
        </div>
      </div>

      {/* Right Panel */}
      <div style={{
        width: '60%', height: '100%',
        display: 'flex', flexDirection: 'column',
        justifyContent: 'center', alignItems: 'center',
        padding: '80px', background: '#ffffff', overflowY: 'auto'
      }}>
        <div style={{ width: '100%', maxWidth: '400px' }}>

          <div style={{ marginBottom: '40px' }}>
            <h3 style={{ fontSize: '26px', fontWeight: '300', color: '#0f172a', letterSpacing: '-0.5px', marginBottom: '8px' }}>
              {isLogin ? 'Sign in to your account' : 'Create your account'}
            </h3>
            <p style={{ fontSize: '13px', color: '#94a3b8', fontWeight: '300' }}>
              {isLogin ? 'Enter your credentials to continue' : 'Fill in the details below to get started'}
            </p>
          </div>

          {error && (
            <div style={{
              background: '#fef2f2', border: '1px solid #fecaca',
              borderRadius: '6px', padding: '12px 16px', marginBottom: '24px',
              color: '#dc2626', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '8px'
            }}>
              ⚠ {error}
            </div>
          )}

          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', marginBottom: '32px' }}>
            {!isLogin && (
              <div>
                <label style={lbl}>Full Name</label>
                <input style={inp} value={form.username}
                  onFocus={e => e.target.style.borderColor = '#2563eb'}
                  onBlur={e => e.target.style.borderColor = '#e2e8f0'}
                  onChange={e => setForm({ ...form, username: e.target.value })} />
              </div>
            )}
            <div>
              <label style={lbl}>Email Address</label>
              <input style={inp} type="email" value={form.email}
                onFocus={e => e.target.style.borderColor = '#2563eb'}
                onBlur={e => e.target.style.borderColor = '#e2e8f0'}
                onChange={e => setForm({ ...form, email: e.target.value })} />
            </div>
            <div>
              <label style={lbl}>Password</label>
              <div style={{ position: 'relative' }}>
                <input style={{ ...inp, paddingRight: '48px' }}
                  type={showPassword ? 'text' : 'password'}
                  value={form.password}
                  onFocus={e => e.target.style.borderColor = '#2563eb'}
                  onBlur={e => e.target.style.borderColor = '#e2e8f0'}
                  onChange={e => setForm({ ...form, password: e.target.value })}
                  onKeyDown={e => e.key === 'Enter' && handle()} />
                <button onClick={() => setShowPassword(!showPassword)} style={{
                  position: 'absolute', right: '14px', top: '50%',
                  transform: 'translateY(-50%)', background: 'none',
                  border: 'none', cursor: 'pointer', color: '#94a3b8',
                  fontSize: '16px', padding: 0, display: 'flex', alignItems: 'center'
                }}>
                  {showPassword ? '🙈' : '👁️'}
                </button>
              </div>
            </div>
          </div>

          <button onClick={handle} disabled={loading}
            onMouseEnter={e => !loading && (e.currentTarget.style.background = '#2563eb')}
            onMouseLeave={e => !loading && (e.currentTarget.style.background = '#0f172a')}
            style={{
              width: '100%', padding: '16px 24px',
              background: loading ? '#94a3b8' : '#0f172a',
              border: 'none', borderRadius: '6px',
              color: 'white', fontSize: '14px', fontWeight: '600',
              cursor: loading ? 'not-allowed' : 'pointer',
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              transition: 'background 0.25s', letterSpacing: '0.3px'
            }}>
            <span>{loading ? 'Please wait...' : isLogin ? 'Sign In' : 'Create Account'}</span>
            <span style={{ fontSize: '18px' }}>{loading ? '⏳' : '→'}</span>
          </button>

          <p style={{ textAlign: 'center', marginTop: '24px', fontSize: '13px', color: '#94a3b8' }}>
            {isLogin ? "Don't have an account? " : "Already have an account? "}
            <span onClick={() => { setIsLogin(!isLogin); setError(null) }}
              style={{ color: '#2563eb', fontWeight: '700', cursor: 'pointer' }}>
              {isLogin ? 'Create one' : 'Sign in'}
            </span>
          </p>

          <p style={{ fontSize: '11px', color: '#e2e8f0', textAlign: 'center', marginTop: '40px', letterSpacing: '1px' }}>
            GLS UNIVERSITY · CAPSTONE 2025–26
          </p>
        </div>
      </div>
    </div>
  )
}