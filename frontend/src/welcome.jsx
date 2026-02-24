import { useState, useEffect } from 'react'

export default function Welcome({ onLogin, onRegister }) {
  const [loaded, setLoaded] = useState(false)
  useEffect(() => { setTimeout(() => setLoaded(true), 80) }, [])

  const fadeIn = (delay = 0) => ({
    opacity: loaded ? 1 : 0,
    transform: loaded ? 'translateY(0)' : 'translateY(20px)',
    transition: `opacity 0.65s ease ${delay}s, transform 0.65s ease ${delay}s`
  })

  const STATS = [
    ['98%', 'X-Ray Accuracy', 'ResNet-50'],
    ['94.75%', 'MRI Accuracy', 'EfficientNet-B3'],
    ['78%', 'Diabetes Accuracy', 'Custom MLP'],
    ['4', 'Tumour Classes', 'Glioma · Meningioma · Pituitary · None'],
  ]

  const FEATURES = [
    ['🫁', 'Chest X-Ray', 'Pneumonia detection with Grad-CAM attention heatmaps', '#2563eb'],
    ['🧠', 'Brain MRI', '4-class tumour classification with EfficientNet-B3', '#0ea5e9'],
    ['🗄️', 'Patient Records', 'Secure MongoDB storage with full history tracking', '#0284c7'],
  ]

  return (
    <div style={{
      minHeight: '100vh', width: '100%',
      background: '#080d1a',
      fontFamily: "'Inter', 'Helvetica Neue', sans-serif",
      display: 'flex', flexDirection: 'column',
      position: 'relative', overflow: 'hidden'
    }}>

      {/* Ambient glows */}
      <div style={{ position: 'fixed', top: '-15%', left: '-8%', width: '640px', height: '640px', borderRadius: '50%', background: 'radial-gradient(circle, rgba(37,99,235,0.14) 0%, transparent 70%)', pointerEvents: 'none' }} />
      <div style={{ position: 'fixed', bottom: '-15%', right: '-8%', width: '520px', height: '520px', borderRadius: '50%', background: 'radial-gradient(circle, rgba(14,165,233,0.09) 0%, transparent 70%)', pointerEvents: 'none' }} />
      {/* Dot grid */}
      <div style={{ position: 'fixed', inset: 0, backgroundImage: `radial-gradient(rgba(255,255,255,0.03) 1px, transparent 1px)`, backgroundSize: '32px 32px', pointerEvents: 'none' }} />

      {/* Navbar */}
      <nav style={{ position: 'relative', zIndex: 10, display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '22px 48px', borderBottom: '1px solid rgba(255,255,255,0.06)', ...fadeIn(0) }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{ width: '34px', height: '34px', borderRadius: '9px', background: 'linear-gradient(135deg, #2563eb, #0ea5e9)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '16px', boxShadow: '0 2px 12px rgba(37,99,235,0.35)' }}>🏥</div>
          <span style={{ fontSize: '16px', fontWeight: '800', color: 'white', letterSpacing: '-0.3px' }}>MediAI</span>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button onClick={onLogin}
            onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,0.09)'}
            onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
            style={{ padding: '8px 20px', background: 'transparent', border: '1px solid rgba(255,255,255,0.14)', borderRadius: '8px', color: 'rgba(255,255,255,0.8)', fontSize: '13px', fontWeight: '500', cursor: 'pointer', transition: 'all 0.2s' }}>
            Sign In
          </button>
          <button onClick={onRegister}
            onMouseEnter={e => { e.currentTarget.style.background = '#1d4ed8'; e.currentTarget.style.transform = 'translateY(-1px)' }}
            onMouseLeave={e => { e.currentTarget.style.background = '#2563eb'; e.currentTarget.style.transform = 'translateY(0)' }}
            style={{ padding: '8px 20px', background: '#2563eb', border: 'none', borderRadius: '8px', color: 'white', fontSize: '13px', fontWeight: '700', cursor: 'pointer', transition: 'all 0.2s', boxShadow: '0 2px 10px rgba(37,99,235,0.35)' }}>
            Get Started
          </button>
        </div>
      </nav>

      {/* Hero */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '56px 24px 48px', position: 'relative', zIndex: 1, textAlign: 'center' }}>

        {/* Badge */}
        <div style={{ ...fadeIn(0.1), marginBottom: '32px' }}>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', padding: '6px 18px', borderRadius: '100px', background: 'rgba(37,99,235,0.14)', border: '1px solid rgba(37,99,235,0.28)', color: '#93c5fd', fontSize: '12px', fontWeight: '600', letterSpacing: '0.4px' }}>
            <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#60a5fa', display: 'inline-block' }} />
            GLS University Capstone Project 2025–26
          </span>
        </div>

        {/* Heading */}
        <div style={{ ...fadeIn(0.2), marginBottom: '8px' }}>
          <h1 style={{ fontSize: 'clamp(40px, 6vw, 72px)', fontWeight: '900', lineHeight: 1.06, letterSpacing: '-3px', color: 'white', margin: 0 }}>
            AI-Powered
          </h1>
        </div>
        <div style={{ ...fadeIn(0.3), marginBottom: '28px' }}>
          <h1 style={{ fontSize: 'clamp(40px, 6vw, 72px)', fontWeight: '900', lineHeight: 1.06, letterSpacing: '-3px', margin: 0, background: 'linear-gradient(90deg, #60a5fa, #0ea5e9, #38bdf8)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            Clinical Diagnostics
          </h1>
        </div>

        {/* Subtitle */}
        <div style={{ ...fadeIn(0.4), marginBottom: '48px' }}>
          <p style={{ fontSize: '17px', color: 'rgba(255,255,255,0.45)', lineHeight: 1.75, maxWidth: '520px', fontWeight: '400', margin: '0 auto' }}>
            ResNet-50 &amp; EfficientNet-B3 powered analysis for pneumonia detection,
            brain tumour classification, and diabetes risk prediction — with Grad-CAM explainability.
          </p>
        </div>

        {/* CTA */}
        <div style={{ ...fadeIn(0.5), display: 'flex', gap: '12px', marginBottom: '72px', flexWrap: 'wrap', justifyContent: 'center' }}>
          <button onClick={onLogin}
            onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-2px)'; e.currentTarget.style.boxShadow = '0 12px 36px rgba(37,99,235,0.5)' }}
            onMouseLeave={e => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = '0 4px 20px rgba(37,99,235,0.3)' }}
            style={{ padding: '15px 34px', background: '#2563eb', border: 'none', borderRadius: '12px', color: 'white', fontSize: '15px', fontWeight: '700', cursor: 'pointer', transition: 'all 0.25s', boxShadow: '0 4px 20px rgba(37,99,235,0.3)', display: 'flex', alignItems: 'center', gap: '10px' }}>
            Sign In to Dashboard <span style={{ fontSize: '17px' }}>→</span>
          </button>
          <button onClick={onRegister}
            onMouseEnter={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.09)'; e.currentTarget.style.transform = 'translateY(-2px)' }}
            onMouseLeave={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.05)'; e.currentTarget.style.transform = 'translateY(0)' }}
            style={{ padding: '15px 34px', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.14)', borderRadius: '12px', color: 'rgba(255,255,255,0.88)', fontSize: '15px', fontWeight: '600', cursor: 'pointer', transition: 'all 0.25s', display: 'flex', alignItems: 'center', gap: '10px' }}>
            Create Account
          </button>
        </div>

        {/* Stats */}
        <div style={{ ...fadeIn(0.6), display: 'flex', gap: '0', marginBottom: '64px', flexWrap: 'wrap', justifyContent: 'center', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)', borderRadius: '16px', overflow: 'hidden' }}>
          {STATS.map(([val, label, sub], i) => (
            <div key={label} style={{ padding: '22px 32px', textAlign: 'center', borderLeft: i > 0 ? '1px solid rgba(255,255,255,0.07)' : 'none' }}>
              <p style={{ fontSize: '30px', fontWeight: '900', color: 'white', letterSpacing: '-1px', margin: 0 }}>{val}</p>
              <p style={{ fontSize: '12px', color: 'rgba(255,255,255,0.55)', margin: '4px 0 2px', fontWeight: '600' }}>{label}</p>
              <p style={{ fontSize: '11px', color: 'rgba(255,255,255,0.25)', margin: 0 }}>{sub}</p>
            </div>
          ))}
        </div>

        {/* Feature cards */}
        <div style={{ ...fadeIn(0.7), display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '14px', maxWidth: '820px', width: '100%' }}>
          {FEATURES.map(([icon, title, desc, color]) => (
            <div key={title}
              onMouseEnter={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.06)'; e.currentTarget.style.borderColor = 'rgba(255,255,255,0.14)'; e.currentTarget.style.transform = 'translateY(-4px)' }}
              onMouseLeave={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.03)'; e.currentTarget.style.borderColor = 'rgba(255,255,255,0.07)'; e.currentTarget.style.transform = 'translateY(0)' }}
              style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)', borderRadius: '16px', padding: '22px', textAlign: 'left', transition: 'all 0.25s ease', cursor: 'default' }}>
              <div style={{ width: '42px', height: '42px', borderRadius: '10px', background: `${color}20`, border: `1px solid ${color}30`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '20px', marginBottom: '14px' }}>
                {icon}
              </div>
              <p style={{ fontSize: '14px', fontWeight: '700', color: 'white', marginBottom: '6px' }}>{title}</p>
              <p style={{ fontSize: '12px', color: 'rgba(255,255,255,0.38)', lineHeight: 1.65, margin: 0 }}>{desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Footer */}
      <div style={{ position: 'relative', zIndex: 1, borderTop: '1px solid rgba(255,255,255,0.06)', padding: '18px 48px', textAlign: 'center', ...fadeIn(0.8) }}>
        <p style={{ fontSize: '11px', color: 'rgba(255,255,255,0.18)', letterSpacing: '1.2px' }}>
          MEDIAI DIAGNOSTICS · GLS UNIVERSITY · INTEGRATED MSC(IT) · CAPSTONE 2025–26
        </p>
      </div>
    </div>
  )
}