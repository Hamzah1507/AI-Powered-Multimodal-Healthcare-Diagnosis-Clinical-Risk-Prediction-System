export default function Welcome({ onLogin, onRegister }) {
  return (
    <div style={{
      minHeight: '100vh',
      background: '#ffffff',
      display: 'flex',
      fontFamily: "'Georgia', 'Times New Roman', serif",
    }}>
      {/* Left Panel */}
      <div style={{
        width: '50%',
        background: 'linear-gradient(160deg, #f0f4ff 0%, #e8f0fe 100%)',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        padding: '80px',
        position: 'relative',
        overflow: 'hidden'
      }}>
        {/* Subtle grid pattern */}
        <div style={{
          position: 'absolute', inset: 0,
          backgroundImage: 'radial-gradient(circle, #2563eb15 1px, transparent 1px)',
          backgroundSize: '32px 32px'
        }} />

        {/* Decorative circle */}
        <div style={{
          position: 'absolute', bottom: '-80px', right: '-80px',
          width: '320px', height: '320px', borderRadius: '50%',
          background: 'linear-gradient(135deg, #2563eb18, #0ea5e918)',
          border: '1px solid #2563eb10'
        }} />
        <div style={{
          position: 'absolute', top: '40px', right: '60px',
          width: '120px', height: '120px', borderRadius: '50%',
          background: 'linear-gradient(135deg, #2563eb10, transparent)',
        }} />

        <div style={{ position: 'relative', zIndex: 1 }}>
          {/* Logo */}
          <div style={{ marginBottom: '48px' }}>
            <span style={{
              fontSize: '13px', fontWeight: '700', letterSpacing: '4px',
              color: '#2563eb', textTransform: 'uppercase',
              fontFamily: "'Helvetica Neue', sans-serif"
            }}>MediAI</span>
          </div>

          <h1 style={{
            fontSize: '52px', fontWeight: '300', lineHeight: 1.15,
            color: '#0f172a', marginBottom: '24px', letterSpacing: '-1px'
          }}>
            Clinical AI<br />
            <span style={{ fontWeight: '700', color: '#2563eb' }}>Diagnostics</span>
          </h1>

          <p style={{
            fontSize: '17px', color: '#64748b', lineHeight: 1.7,
            marginBottom: '48px', maxWidth: '380px',
            fontFamily: "'Helvetica Neue', sans-serif", fontWeight: '300'
          }}>
            AI-powered medical imaging analysis for pneumonia detection, brain tumor classification, and diabetes risk prediction.
          </p>

          {/* Stats row */}
          <div style={{ display: 'flex', gap: '40px' }}>
            {[['98%', 'X-Ray Accuracy'], ['94.75%', 'MRI Accuracy'], ['4', 'Tumor Classes']].map(([val, label]) => (
              <div key={label}>
                <p style={{
                  fontSize: '26px', fontWeight: '700', color: '#0f172a',
                  fontFamily: "'Helvetica Neue', sans-serif", letterSpacing: '-0.5px'
                }}>{val}</p>
                <p style={{
                  fontSize: '11px', color: '#94a3b8', marginTop: '2px',
                  fontFamily: "'Helvetica Neue', sans-serif",
                  textTransform: 'uppercase', letterSpacing: '1px'
                }}>{label}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Right Panel */}
      <div style={{
        width: '50%',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        padding: '80px',
        background: '#ffffff'
      }}>
        <div style={{ width: '100%', maxWidth: '380px' }}>

          {/* Header */}
          <div style={{ marginBottom: '48px' }}>
            <h2 style={{
              fontSize: '28px', fontWeight: '300', color: '#0f172a',
              marginBottom: '8px', letterSpacing: '-0.5px'
            }}>Welcome back</h2>
            <p style={{
              fontSize: '14px', color: '#94a3b8',
              fontFamily: "'Helvetica Neue', sans-serif"
            }}>Sign in to access your clinical dashboard</p>
          </div>

          {/* Login Button */}
          <button onClick={onLogin} style={{
            width: '100%', padding: '16px 24px',
            background: '#0f172a', border: 'none', borderRadius: '4px',
            color: 'white', fontSize: '14px', fontWeight: '500',
            cursor: 'pointer', letterSpacing: '0.5px', marginBottom: '12px',
            fontFamily: "'Helvetica Neue', sans-serif",
            transition: 'all 0.2s',
          }}
            onMouseEnter={e => e.target.style.background = '#2563eb'}
            onMouseLeave={e => e.target.style.background = '#0f172a'}
          >
            Sign In
          </button>

          {/* Divider */}
          <div style={{
            display: 'flex', alignItems: 'center', gap: '16px',
            margin: '24px 0'
          }}>
            <div style={{ flex: 1, height: '1px', background: '#f1f5f9' }} />
            <span style={{
              fontSize: '12px', color: '#cbd5e1',
              fontFamily: "'Helvetica Neue', sans-serif"
            }}>or</span>
            <div style={{ flex: 1, height: '1px', background: '#f1f5f9' }} />
          </div>

          {/* Register Button */}
          <button onClick={onRegister} style={{
            width: '100%', padding: '16px 24px',
            background: 'white', border: '1px solid #e2e8f0', borderRadius: '4px',
            color: '#0f172a', fontSize: '14px', fontWeight: '500',
            cursor: 'pointer', letterSpacing: '0.5px',
            fontFamily: "'Helvetica Neue', sans-serif",
            transition: 'all 0.2s'
          }}
            onMouseEnter={e => { e.target.style.borderColor = '#2563eb'; e.target.style.color = '#2563eb' }}
            onMouseLeave={e => { e.target.style.borderColor = '#e2e8f0'; e.target.style.color = '#0f172a' }}
          >
            Create Account
          </button>

          {/* Features */}
          <div style={{ marginTop: '48px', paddingTop: '32px', borderTop: '1px solid #f8fafc' }}>
            {[
              ['Chest X-Ray Analysis', 'Pneumonia detection with Grad-CAM'],
              ['Brain MRI Scanning', '4-class tumor classification'],
              ['Patient Records', 'Secure MongoDB storage']
            ].map(([title, desc]) => (
              <div key={title} style={{
                display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px'
              }}>
                <div style={{
                  width: '6px', height: '6px', borderRadius: '50%',
                  background: '#2563eb', flexShrink: 0
                }} />
                <div>
                  <p style={{
                    fontSize: '13px', fontWeight: '600', color: '#0f172a',
                    fontFamily: "'Helvetica Neue', sans-serif"
                  }}>{title}</p>
                  <p style={{
                    fontSize: '12px', color: '#94a3b8',
                    fontFamily: "'Helvetica Neue', sans-serif"
                  }}>{desc}</p>
                </div>
              </div>
            ))}
          </div>

          {/* Footer */}
          <p style={{
            marginTop: '32px', fontSize: '11px', color: '#cbd5e1',
            fontFamily: "'Helvetica Neue', sans-serif",
            textAlign: 'center', letterSpacing: '0.5px'
          }}>
            GLS University · Capstone 2025–26
          </p>
        </div>
      </div>
    </div>
  )
}