export default function Welcome({ onLogin, onRegister }) {
  return (
    <div style={{
      height: '100vh', width: '100vw', overflow: 'hidden',
      background: '#ffffff',
      display: 'flex',
      fontFamily: "'Helvetica Neue', Arial, sans-serif",
    }}>
      {/* Left Panel */}
      <div style={{
        width: '55%', height: '100%',
        background: 'linear-gradient(145deg, #eef2ff 0%, #e0e9ff 60%, #dbeafe 100%)',
        display: 'flex', flexDirection: 'column',
        justifyContent: 'center', padding: '80px',
        position: 'relative', overflow: 'hidden'
      }}>
        {/* Grid pattern */}
        <div style={{
          position: 'absolute', inset: 0,
          backgroundImage: 'radial-gradient(circle, #2563eb12 1px, transparent 1px)',
          backgroundSize: '28px 28px'
        }} />
        {/* Circles */}
        <div style={{
          position: 'absolute', bottom: '-120px', right: '-60px',
          width: '420px', height: '420px', borderRadius: '50%',
          background: 'radial-gradient(circle, #2563eb14 0%, transparent 70%)'
        }} />
        <div style={{
          position: 'absolute', top: '-60px', left: '-60px',
          width: '280px', height: '280px', borderRadius: '50%',
          background: 'radial-gradient(circle, #6366f114 0%, transparent 70%)'
        }} />

        <div style={{ position: 'relative', zIndex: 1 }}>
          {/* Brand */}
          <div style={{ marginBottom: '56px' }}>
            <span style={{
              fontSize: '11px', fontWeight: '800', letterSpacing: '5px',
              color: '#2563eb', textTransform: 'uppercase',
            }}>MediAI</span>
            <div style={{ width: '32px', height: '2px', background: '#2563eb', marginTop: '8px' }} />
          </div>

          <h1 style={{
            fontSize: '58px', fontWeight: '300', lineHeight: 1.1,
            color: '#0f172a', marginBottom: '6px', letterSpacing: '-2px'
          }}>Clinical AI</h1>
          <h1 style={{
            fontSize: '58px', fontWeight: '800', lineHeight: 1.1,
            color: '#2563eb', marginBottom: '28px', letterSpacing: '-2px'
          }}>Diagnostics</h1>

          <p style={{
            fontSize: '16px', color: '#64748b', lineHeight: 1.8,
            marginBottom: '56px', maxWidth: '400px', fontWeight: '300'
          }}>
            AI-powered medical imaging for pneumonia detection, brain tumor classification, and diabetes risk prediction.
          </p>

          {/* Stats */}
          <div style={{ display: 'flex', gap: '48px' }}>
            {[['98%', 'X-Ray Accuracy'], ['94.75%', 'MRI Accuracy'], ['78%', 'Diabetes Accuracy']].map(([val, label]) => (
              <div key={label}>
                <p style={{ fontSize: '28px', fontWeight: '800', color: '#0f172a', letterSpacing: '-1px' }}>{val}</p>
                <p style={{ fontSize: '10px', color: '#94a3b8', marginTop: '4px', textTransform: 'uppercase', letterSpacing: '1.5px' }}>{label}</p>
              </div>
            ))}
          </div>

          {/* Tags */}
          <div style={{ display: 'flex', gap: '8px', marginTop: '40px', flexWrap: 'wrap' }}>
            {['ResNet-50', 'EfficientNet-B3', 'Grad-CAM', 'MongoDB', 'FastAPI'].map(tag => (
              <span key={tag} style={{
                padding: '5px 12px', borderRadius: '100px',
                background: 'rgba(37,99,235,0.08)', border: '1px solid rgba(37,99,235,0.15)',
                color: '#2563eb', fontSize: '11px', fontWeight: '600', letterSpacing: '0.3px'
              }}>{tag}</span>
            ))}
          </div>
        </div>
      </div>

      {/* Right Panel */}
      <div style={{
        width: '45%', height: '100%',
        display: 'flex', flexDirection: 'column',
        justifyContent: 'center', alignItems: 'center',
        padding: '80px', background: '#ffffff'
      }}>
        <div style={{ width: '100%', maxWidth: '360px' }}>

          <div style={{ marginBottom: '52px' }}>
            <h2 style={{
              fontSize: '32px', fontWeight: '300', color: '#0f172a',
              marginBottom: '10px', letterSpacing: '-1px'
            }}>Welcome back</h2>
            <p style={{ fontSize: '14px', color: '#94a3b8', fontWeight: '300' }}>
              Sign in to access your clinical dashboard
            </p>
          </div>

          {/* Sign In Button */}
          <button onClick={onLogin}
            onMouseEnter={e => { e.currentTarget.style.background = '#2563eb'; e.currentTarget.querySelector('.arr').style.transform = 'translateX(4px)' }}
            onMouseLeave={e => { e.currentTarget.style.background = '#0f172a'; e.currentTarget.querySelector('.arr').style.transform = 'translateX(0)' }}
            style={{
              width: '100%', padding: '17px 24px',
              background: '#0f172a', border: 'none', borderRadius: '6px',
              color: 'white', fontSize: '14px', fontWeight: '600',
              cursor: 'pointer', letterSpacing: '0.3px', marginBottom: '12px',
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              transition: 'background 0.25s'
            }}>
            <span>Sign In</span>
            <span className="arr" style={{ fontSize: '18px', transition: 'transform 0.2s' }}>→</span>
          </button>

          {/* Divider */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px', margin: '20px 0' }}>
            <div style={{ flex: 1, height: '1px', background: '#f1f5f9' }} />
            <span style={{ fontSize: '11px', color: '#cbd5e1', letterSpacing: '1px' }}>OR</span>
            <div style={{ flex: 1, height: '1px', background: '#f1f5f9' }} />
          </div>

          {/* Create Account Button */}
          <button onClick={onRegister}
            onMouseEnter={e => { e.currentTarget.style.borderColor = '#2563eb'; e.currentTarget.style.color = '#2563eb'; e.currentTarget.querySelector('.arr').style.transform = 'translateX(4px)' }}
            onMouseLeave={e => { e.currentTarget.style.borderColor = '#e2e8f0'; e.currentTarget.style.color = '#0f172a'; e.currentTarget.querySelector('.arr').style.transform = 'translateX(0)' }}
            style={{
              width: '100%', padding: '17px 24px',
              background: 'white', border: '1.5px solid #e2e8f0', borderRadius: '6px',
              color: '#0f172a', fontSize: '14px', fontWeight: '600',
              cursor: 'pointer', letterSpacing: '0.3px',
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              transition: 'all 0.25s'
            }}>
            <span>Create Account</span>
            <span className="arr" style={{ fontSize: '18px', transition: 'transform 0.2s' }}>→</span>
          </button>

          {/* Features */}
          <div style={{ marginTop: '52px', paddingTop: '32px', borderTop: '1px solid #f8fafc' }}>
            {[
              ['Chest X-Ray', 'Pneumonia detection with Grad-CAM heatmaps'],
              ['Brain MRI', '4-class tumor classification'],
              ['Patient Records', 'Secure MongoDB storage & history'],
            ].map(([title, desc]) => (
              <div key={title} style={{ display: 'flex', gap: '14px', marginBottom: '20px', alignItems: 'flex-start' }}>
                <div style={{ width: '5px', height: '5px', borderRadius: '50%', background: '#2563eb', marginTop: '6px', flexShrink: 0 }} />
                <div>
                  <p style={{ fontSize: '13px', fontWeight: '700', color: '#0f172a' }}>{title}</p>
                  <p style={{ fontSize: '12px', color: '#94a3b8', marginTop: '2px' }}>{desc}</p>
                </div>
              </div>
            ))}
          </div>

          <p style={{ fontSize: '11px', color: '#e2e8f0', textAlign: 'center', marginTop: '24px', letterSpacing: '1px' }}>
            GLS UNIVERSITY · CAPSTONE 2025–26
          </p>
        </div>
      </div>
    </div>
  )
}