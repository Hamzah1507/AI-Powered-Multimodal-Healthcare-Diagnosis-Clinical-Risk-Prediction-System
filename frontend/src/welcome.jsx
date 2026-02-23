export default function Welcome({ onLogin, onRegister }) {
  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #0f172a 100%)',
      display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center',
      fontFamily: '-apple-system, BlinkMacSystemFont, sans-serif'
    }}>

      {/* Animated background dots */}
      <div style={{ position: 'absolute', inset: 0, overflow: 'hidden', zIndex: 0 }}>
        {[...Array(20)].map((_, i) => (
          <div key={i} style={{
            position: 'absolute',
            width: `${Math.random() * 6 + 2}px`,
            height: `${Math.random() * 6 + 2}px`,
            borderRadius: '50%',
            background: 'rgba(255,255,255,0.1)',
            left: `${Math.random() * 100}%`,
            top: `${Math.random() * 100}%`,
          }} />
        ))}
      </div>

      {/* Main Card */}
      <div style={{
        position: 'relative', zIndex: 1,
        background: 'rgba(255,255,255,0.05)',
        backdropFilter: 'blur(20px)',
        border: '1px solid rgba(255,255,255,0.1)',
        borderRadius: '24px', padding: '56px 64px',
        textAlign: 'center', maxWidth: '520px', width: '90%',
        boxShadow: '0 25px 50px rgba(0,0,0,0.5)'
      }}>

        {/* Logo */}
        <div style={{
          background: 'linear-gradient(135deg, #2563eb, #0ea5e9)',
          borderRadius: '20px', padding: '16px',
          fontSize: '40px', display: 'inline-block',
          marginBottom: '24px',
          boxShadow: '0 8px 32px rgba(37,99,235,0.4)'
        }}>🏥</div>

        <h1 style={{
          fontSize: '32px', fontWeight: '900',
          color: 'white', marginBottom: '8px', letterSpacing: '-0.5px'
        }}>MediAI Diagnostics</h1>

        <p style={{
          color: 'rgba(255,255,255,0.6)', fontSize: '15px',
          marginBottom: '8px'
        }}>AI-Powered Clinical Decision Support</p>

        <p style={{
          color: 'rgba(255,255,255,0.4)', fontSize: '12px',
          marginBottom: '40px'
        }}>GLS University Capstone Project 2025–26</p>

        {/* Feature Pills */}
        <div style={{
          display: 'flex', flexWrap: 'wrap', gap: '8px',
          justifyContent: 'center', marginBottom: '40px'
        }}>
          {[
            ['🫁', 'Chest X-Ray'],
            ['🧠', 'Brain MRI'],
            ['🔥', 'Grad-CAM'],
            ['📄', 'PDF Reports'],
            ['🗄️', 'Patient History']
          ].map(([icon, label]) => (
            <div key={label} style={{
              background: 'rgba(255,255,255,0.08)',
              border: '1px solid rgba(255,255,255,0.12)',
              borderRadius: '20px', padding: '6px 14px',
              color: 'rgba(255,255,255,0.8)', fontSize: '13px',
              display: 'flex', alignItems: 'center', gap: '6px'
            }}>
              <span>{icon}</span> {label}
            </div>
          ))}
        </div>

        {/* Buttons */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <button onClick={onLogin} style={{
            width: '100%', padding: '16px',
            background: 'linear-gradient(135deg, #2563eb, #0ea5e9)',
            border: 'none', borderRadius: '12px',
            color: 'white', fontSize: '16px', fontWeight: '700',
            cursor: 'pointer',
            boxShadow: '0 4px 20px rgba(37,99,235,0.4)',
            transition: 'all 0.2s'
          }}>
            🔐 Login to Dashboard
          </button>

          <button onClick={onRegister} style={{
            width: '100%', padding: '16px',
            background: 'transparent',
            border: '1.5px solid rgba(255,255,255,0.2)',
            borderRadius: '12px',
            color: 'white', fontSize: '16px', fontWeight: '700',
            cursor: 'pointer', transition: 'all 0.2s'
          }}>
            ✨ Create New Account
          </button>
        </div>

        {/* Footer */}
        <p style={{
          color: 'rgba(255,255,255,0.3)', fontSize: '12px',
          marginTop: '32px'
        }}>
          Integrated MSc(IT) • Sahana System Limited
        </p>
      </div>
    </div>
  )
}