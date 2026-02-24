import { useState, useEffect } from 'react'

export default function Welcome({ onLogin, onRegister }) {
  const [loaded, setLoaded] = useState(false)

  useEffect(() => {
    setTimeout(() => setLoaded(true), 100)
  }, [])

  const fadeIn = (delay = 0) => ({
    opacity: loaded ? 1 : 0,
    transform: loaded ? 'translateY(0)' : 'translateY(24px)',
    transition: `opacity 0.7s ease ${delay}s, transform 0.7s ease ${delay}s`
  })

  return (
    <div style={{
      minHeight: '100vh', width: '100%',
      background: '#0a0f1e',
      fontFamily: "'DM Sans', 'Helvetica Neue', sans-serif",
      display: 'flex', flexDirection: 'column',
      position: 'relative', overflow: 'hidden'
    }}>

      {/* Ambient glow effects */}
      <div style={{
        position: 'fixed', top: '-20%', left: '-10%',
        width: '600px', height: '600px', borderRadius: '50%',
        background: 'radial-gradient(circle, rgba(37,99,235,0.15) 0%, transparent 70%)',
        pointerEvents: 'none'
      }} />
      <div style={{
        position: 'fixed', bottom: '-20%', right: '-10%',
        width: '500px', height: '500px', borderRadius: '50%',
        background: 'radial-gradient(circle, rgba(14,165,233,0.1) 0%, transparent 70%)',
        pointerEvents: 'none'
      }} />

      {/* Subtle grid */}
      <div style={{
        position: 'fixed', inset: 0,
        backgroundImage: `linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
                          linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px)`,
        backgroundSize: '60px 60px',
        pointerEvents: 'none'
      }} />

      {/* Navbar */}
      <nav style={{
        position: 'relative', zIndex: 10,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '24px 48px',
        borderBottom: '1px solid rgba(255,255,255,0.06)',
        ...fadeIn(0)
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            width: '32px', height: '32px', borderRadius: '8px',
            background: 'linear-gradient(135deg, #2563eb, #0ea5e9)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: '16px', fontWeight: '900', color: 'white'
          }}>+</div>
          <span style={{ fontSize: '16px', fontWeight: '700', color: 'white', letterSpacing: '-0.3px' }}>MediAI</span>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button onClick={onLogin}
            onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,0.1)'}
            onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
            style={{
              padding: '8px 20px', background: 'transparent',
              border: '1px solid rgba(255,255,255,0.15)', borderRadius: '8px',
              color: 'rgba(255,255,255,0.8)', fontSize: '13px', fontWeight: '500',
              cursor: 'pointer', transition: 'all 0.2s'
            }}>Sign In</button>
          <button onClick={onRegister}
            onMouseEnter={e => { e.currentTarget.style.background = '#1d4ed8'; e.currentTarget.style.transform = 'translateY(-1px)' }}
            onMouseLeave={e => { e.currentTarget.style.background = '#2563eb'; e.currentTarget.style.transform = 'translateY(0)' }}
            style={{
              padding: '8px 20px', background: '#2563eb',
              border: 'none', borderRadius: '8px',
              color: 'white', fontSize: '13px', fontWeight: '600',
              cursor: 'pointer', transition: 'all 0.2s'
            }}>Get Started</button>
        </div>
      </nav>

      {/* Hero */}
      <div style={{
        flex: 1, display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center',
        padding: '60px 24px', position: 'relative', zIndex: 1, textAlign: 'center'
      }}>

        {/* Badge */}
        <div style={{ ...fadeIn(0.1), marginBottom: '28px' }}>
          <span style={{
            display: 'inline-flex', alignItems: 'center', gap: '8px',
            padding: '6px 16px', borderRadius: '100px',
            background: 'rgba(37,99,235,0.15)',
            border: '1px solid rgba(37,99,235,0.3)',
            color: '#60a5fa', fontSize: '12px', fontWeight: '600', letterSpacing: '0.5px'
          }}>
            <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#60a5fa', display: 'inline-block' }} />
            GLS University Capstone Project 2025–26
          </span>
        </div>

        {/* Main heading */}
        <div style={{ ...fadeIn(0.2), marginBottom: '12px' }}>
          <h1 style={{
            fontSize: 'clamp(42px, 6vw, 76px)', fontWeight: '800',
            lineHeight: 1.05, letterSpacing: '-3px', color: 'white',
            margin: 0
          }}>
            AI-Powered
          </h1>
        </div>
        <div style={{ ...fadeIn(0.3), marginBottom: '28px' }}>
          <h1 style={{
            fontSize: 'clamp(42px, 6vw, 76px)', fontWeight: '800',
            lineHeight: 1.05, letterSpacing: '-3px', margin: 0,
            background: 'linear-gradient(90deg, #3b82f6, #0ea5e9, #38bdf8)',
            WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent'
          }}>
            Clinical Diagnostics
          </h1>
        </div>

        <div style={{ ...fadeIn(0.4), marginBottom: '48px' }}>
          <p style={{
            fontSize: '18px', color: 'rgba(255,255,255,0.5)',
            lineHeight: 1.7, maxWidth: '540px', fontWeight: '300', margin: '0 auto'
          }}>
            ResNet-50 & EfficientNet-B3 powered analysis for pneumonia detection,
            brain tumor classification, and diabetes risk prediction — with Grad-CAM explainability.
          </p>
        </div>

        {/* CTA Buttons */}
        <div style={{ ...fadeIn(0.5), display: 'flex', gap: '12px', marginBottom: '72px', flexWrap: 'wrap', justifyContent: 'center' }}>
          <button onClick={onLogin}
            onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-2px)'; e.currentTarget.style.boxShadow = '0 12px 40px rgba(37,99,235,0.5)' }}
            onMouseLeave={e => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = '0 4px 20px rgba(37,99,235,0.3)' }}
            style={{
              padding: '16px 36px', background: '#2563eb',
              border: 'none', borderRadius: '12px',
              color: 'white', fontSize: '15px', fontWeight: '700',
              cursor: 'pointer', transition: 'all 0.25s ease',
              boxShadow: '0 4px 20px rgba(37,99,235,0.3)',
              display: 'flex', alignItems: 'center', gap: '10px'
            }}>
            Sign In to Dashboard <span style={{ fontSize: '18px' }}>→</span>
          </button>
          <button onClick={onRegister}
            onMouseEnter={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.1)'; e.currentTarget.style.transform = 'translateY(-2px)' }}
            onMouseLeave={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.05)'; e.currentTarget.style.transform = 'translateY(0)' }}
            style={{
              padding: '16px 36px',
              background: 'rgba(255,255,255,0.05)',
              border: '1px solid rgba(255,255,255,0.15)', borderRadius: '12px',
              color: 'rgba(255,255,255,0.9)', fontSize: '15px', fontWeight: '600',
              cursor: 'pointer', transition: 'all 0.25s ease',
              display: 'flex', alignItems: 'center', gap: '10px'
            }}>
            Create Account
          </button>
        </div>

        {/* Stats */}
        <div style={{ ...fadeIn(0.6), display: 'flex', gap: '0', marginBottom: '64px', flexWrap: 'wrap', justifyContent: 'center' }}>
          {[
            ['98%', 'X-Ray Accuracy', 'ResNet-50'],
            ['94.75%', 'MRI Accuracy', 'EfficientNet-B3'],
            ['78%', 'Diabetes Accuracy', 'Custom MLP'],
            ['4', 'Tumor Classes', 'Glioma · Meningioma · Pituitary · None'],
          ].map(([val, label, sub], i) => (
            <div key={label} style={{
              padding: '24px 36px', textAlign: 'center',
              borderLeft: i > 0 ? '1px solid rgba(255,255,255,0.08)' : 'none'
            }}>
              <p style={{ fontSize: '32px', fontWeight: '800', color: 'white', letterSpacing: '-1px', margin: 0 }}>{val}</p>
              <p style={{ fontSize: '13px', color: 'rgba(255,255,255,0.6)', margin: '4px 0 2px', fontWeight: '600' }}>{label}</p>
              <p style={{ fontSize: '11px', color: 'rgba(255,255,255,0.3)', margin: 0 }}>{sub}</p>
            </div>
          ))}
        </div>

        {/* Feature Cards */}
        <div style={{ ...fadeIn(0.7), display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px', maxWidth: '800px', width: '100%' }}>
          {[
            ['🫁', 'Chest X-Ray', 'Pneumonia detection with Grad-CAM attention heatmaps', '#2563eb'],
            ['🧠', 'Brain MRI', '4-class tumor classification with EfficientNet-B3', '#0ea5e9'],
            ['🗄️', 'Patient Records', 'Secure MongoDB storage with full history tracking', '#0284c7'],
          ].map(([icon, title, desc, color]) => (
            <div key={title}
              onMouseEnter={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.07)'; e.currentTarget.style.borderColor = 'rgba(255,255,255,0.15)'; e.currentTarget.style.transform = 'translateY(-4px)' }}
              onMouseLeave={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.03)'; e.currentTarget.style.borderColor = 'rgba(255,255,255,0.08)'; e.currentTarget.style.transform = 'translateY(0)' }}
              style={{
                background: 'rgba(255,255,255,0.03)',
                border: '1px solid rgba(255,255,255,0.08)',
                borderRadius: '16px', padding: '24px',
                textAlign: 'left', cursor: 'default',
                transition: 'all 0.25s ease'
              }}>
              <div style={{
                width: '40px', height: '40px', borderRadius: '10px',
                background: `${color}22`, border: `1px solid ${color}33`,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: '20px', marginBottom: '14px'
              }}>{icon}</div>
              <p style={{ fontSize: '14px', fontWeight: '700', color: 'white', marginBottom: '6px' }}>{title}</p>
              <p style={{ fontSize: '12px', color: 'rgba(255,255,255,0.4)', lineHeight: 1.6, margin: 0 }}>{desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Bottom footer */}
      <div style={{
        position: 'relative', zIndex: 1,
        borderTop: '1px solid rgba(255,255,255,0.06)',
        padding: '20px 48px',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        ...fadeIn(0.8)
      }}>
        <p style={{ fontSize: '12px', color: 'rgba(255,255,255,0.2)', letterSpacing: '1px' }}>
          MEDIAI DIAGNOSTICS · GLS UNIVERSITY · INTEGRATED MSC(IT) · CAPSTONE 2025–26
        </p>
      </div>
    </div>
  )
}