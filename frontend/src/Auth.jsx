import { useState, useEffect } from 'react'
import axios from 'axios'

const API = 'http://127.0.0.1:8000'

export default function Auth({ mode, onSuccess, onBack }) {
  const [form, setForm] = useState({ username: '', email: '', password: '' })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [isLogin, setIsLogin] = useState(mode === 'login')
  const [showPassword, setShowPassword] = useState(false)
  const [loaded, setLoaded] = useState(false)

  useEffect(() => { setTimeout(() => setLoaded(true), 100) }, [])

  const fadeIn = (delay = 0) => ({
    opacity: loaded ? 1 : 0,
    transform: loaded ? 'translateY(0)' : 'translateY(20px)',
    transition: `opacity 0.6s ease ${delay}s, transform 0.6s ease ${delay}s`
  })

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
    border: '1px solid rgba(255,255,255,0.1)',
    borderRadius: '10px',
    background: 'rgba(255,255,255,0.05)',
    color: 'white', fontSize: '14px',
    boxSizing: 'border-box', outline: 'none',
    fontFamily: "'Helvetica Neue', Arial, sans-serif",
    transition: 'border-color 0.2s, background 0.2s'
  }

  const lbl = {
    fontSize: '11px', fontWeight: '700', color: 'rgba(255,255,255,0.4)',
    textTransform: 'uppercase', letterSpacing: '1.5px',
    display: 'block', marginBottom: '8px'
  }

  return (
    <div style={{
      minHeight: '100vh', width: '100%',
      background: '#0a0f1e',
      fontFamily: "'Helvetica Neue', Arial, sans-serif",
      display: 'flex', flexDirection: 'column',
      position: 'relative', overflow: 'hidden'
    }}>
      {/* Ambient glows */}
      <div style={{
        position: 'fixed', top: '-20%', left: '-10%',
        width: '500px', height: '500px', borderRadius: '50%',
        background: 'radial-gradient(circle, rgba(37,99,235,0.12) 0%, transparent 70%)',
        pointerEvents: 'none'
      }} />
      <div style={{
        position: 'fixed', bottom: '-20%', right: '-10%',
        width: '400px', height: '400px', borderRadius: '50%',
        background: 'radial-gradient(circle, rgba(14,165,233,0.08) 0%, transparent 70%)',
        pointerEvents: 'none'
      }} />
      {/* Grid */}
      <div style={{
        position: 'fixed', inset: 0,
        backgroundImage: `linear-gradient(rgba(255,255,255,0.015) 1px, transparent 1px),
                          linear-gradient(90deg, rgba(255,255,255,0.015) 1px, transparent 1px)`,
        backgroundSize: '60px 60px', pointerEvents: 'none'
      }} />

      {/* Top bar with back button */}
      <div style={{
        position: 'relative', zIndex: 10,
        padding: '24px 48px',
        borderBottom: '1px solid rgba(255,255,255,0.06)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        ...fadeIn(0)
      }}>
        {/* Back button */}
        <button onClick={onBack}
          onMouseEnter={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.08)'; e.currentTarget.style.borderColor = 'rgba(255,255,255,0.2)' }}
          onMouseLeave={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.04)'; e.currentTarget.style.borderColor = 'rgba(255,255,255,0.1)' }}
          style={{
            display: 'flex', alignItems: 'center', gap: '8px',
            padding: '8px 16px', borderRadius: '8px',
            background: 'rgba(255,255,255,0.04)',
            border: '1px solid rgba(255,255,255,0.1)',
            color: 'rgba(255,255,255,0.6)', fontSize: '13px', fontWeight: '500',
            cursor: 'pointer', transition: 'all 0.2s'
          }}>
          ← Back
        </button>

        {/* Brand */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            width: '32px', height: '32px', borderRadius: '8px',
            background: 'linear-gradient(135deg, #2563eb, #0ea5e9)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: '16px', fontWeight: '900', color: 'white'
          }}>+</div>
          <span style={{ fontSize: '16px', fontWeight: '700', color: 'white' }}>MediAI</span>
        </div>
      </div>

      {/* Main content */}
      <div style={{
        flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
        padding: '16px 24px', position: 'relative', zIndex: 1
      }}>
        <div style={{ width: '100%', maxWidth: '420px' }}>

          {/* Header */}
          <div style={{ textAlign: 'center', marginBottom: '20px', ...fadeIn(0.1) }}>
            <div style={{
              width: '52px', height: '52px', borderRadius: '14px',
              background: 'linear-gradient(135deg, #2563eb, #0ea5e9)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              margin: '0 auto 20px', fontSize: '22px', fontWeight: '900', color: 'white'
            }}>+</div>
            <h2 style={{
              fontSize: '26px', fontWeight: '700', color: 'white',
              letterSpacing: '-0.5px', marginBottom: '8px'
            }}>
              {isLogin ? 'Welcome back' : 'Create your account'}
            </h2>
            <p style={{ fontSize: '14px', color: 'rgba(255,255,255,0.4)', fontWeight: '300' }}>
              {isLogin ? 'Sign in to access your clinical dashboard' : 'Join MediAI to get started'}
            </p>
          </div>

          {/* Card */}
          <div style={{
            background: 'rgba(255,255,255,0.04)',
            border: '1px solid rgba(255,255,255,0.08)',
            borderRadius: '16px', padding: '24px',
            ...fadeIn(0.2)
          }}>

            {/* Error */}
            {error && (
              <div style={{
                background: 'rgba(220,38,38,0.15)', border: '1px solid rgba(220,38,38,0.3)',
                borderRadius: '8px', padding: '12px 16px', marginBottom: '20px',
                color: '#fca5a5', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '8px'
              }}>⚠ {error}</div>
            )}

            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginBottom: '16px' }}>
              {!isLogin && (
                <div>
                  <label style={lbl}>Full Name</label>
                  <input style={inp} value={form.username}
                    onFocus={e => { e.target.style.borderColor = '#2563eb'; e.target.style.background = 'rgba(37,99,235,0.08)' }}
                    onBlur={e => { e.target.style.borderColor = 'rgba(255,255,255,0.1)'; e.target.style.background = 'rgba(255,255,255,0.05)' }}
                    onChange={e => setForm({ ...form, username: e.target.value })} />
                </div>
              )}
              <div>
                <label style={lbl}>Email Address</label>
                <input style={inp} type="email" value={form.email}
                  onFocus={e => { e.target.style.borderColor = '#2563eb'; e.target.style.background = 'rgba(37,99,235,0.08)' }}
                  onBlur={e => { e.target.style.borderColor = 'rgba(255,255,255,0.1)'; e.target.style.background = 'rgba(255,255,255,0.05)' }}
                  onChange={e => setForm({ ...form, email: e.target.value })} />
              </div>
              <div>
                <label style={lbl}>Password</label>
                <div style={{ position: 'relative' }}>
                  <input style={{ ...inp, paddingRight: '48px' }}
                    type={showPassword ? 'text' : 'password'}
                    value={form.password}
                    onFocus={e => { e.target.style.borderColor = '#2563eb'; e.target.style.background = 'rgba(37,99,235,0.08)' }}
                    onBlur={e => { e.target.style.borderColor = 'rgba(255,255,255,0.1)'; e.target.style.background = 'rgba(255,255,255,0.05)' }}
                    onChange={e => setForm({ ...form, password: e.target.value })}
                    onKeyDown={e => e.key === 'Enter' && handle()} />
                  <button onClick={() => setShowPassword(!showPassword)} style={{
                    position: 'absolute', right: '14px', top: '50%',
                    transform: 'translateY(-50%)',
                    background: 'none', border: 'none', cursor: 'pointer',
                    color: 'rgba(255,255,255,0.4)', fontSize: '16px', padding: 0,
                    display: 'flex', alignItems: 'center'
                  }}>
                    {showPassword ? '🙈' : '👁️'}
                  </button>
                </div>
              </div>
            </div>

            {/* Submit */}
            <button onClick={handle} disabled={loading}
              onMouseEnter={e => !loading && (e.currentTarget.style.boxShadow = '0 8px 30px rgba(37,99,235,0.5)')}
              onMouseLeave={e => !loading && (e.currentTarget.style.boxShadow = '0 4px 16px rgba(37,99,235,0.3)')}
              style={{
                width: '100%', padding: '15px 24px',
                background: loading ? 'rgba(255,255,255,0.1)' : 'linear-gradient(135deg, #2563eb, #1d4ed8)',
                border: 'none', borderRadius: '10px',
                color: loading ? 'rgba(255,255,255,0.4)' : 'white',
                fontSize: '15px', fontWeight: '700',
                cursor: loading ? 'not-allowed' : 'pointer',
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                transition: 'all 0.2s ease',
                boxShadow: '0 4px 16px rgba(37,99,235,0.3)'
              }}>
              <span>{loading ? 'Please wait...' : isLogin ? 'Sign In' : 'Create Account'}</span>
              <span style={{ fontSize: '20px' }}>{loading ? '⏳' : '→'}</span>
            </button>

            {/* Switch */}
            <p style={{ textAlign: 'center', marginTop: '20px', fontSize: '13px', color: 'rgba(255,255,255,0.35)' }}>
              {isLogin ? "Don't have an account? " : "Already have an account? "}
              <span onClick={() => { setIsLogin(!isLogin); setError(null) }}
                style={{ color: '#60a5fa', fontWeight: '700', cursor: 'pointer' }}>
                {isLogin ? 'Create one' : 'Sign in'}
              </span>
            </p>
          </div>

          <p style={{
            textAlign: 'center', marginTop: '24px',
            fontSize: '11px', color: 'rgba(255,255,255,0.15)', letterSpacing: '1px'
          }}>
            GLS UNIVERSITY · CAPSTONE 2025–26
          </p>
        </div>
      </div>
    </div>
  )
}