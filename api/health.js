// ============================================
// Vercel Serverless Function — GET /api/health
// ============================================
// Endpoint de diagnóstico rápido: confirma que la función está desplegada
// y reporta si la env var crítica está configurada (sin exponer su valor).

export default function handler(_req, res) {
  res.status(200).json({
    ok: true,
    provider: 'openrouter',
    model: process.env.OPENROUTER_MODEL || 'openai/gpt-oss-120b:free',
    apiKeyConfigured: Boolean(process.env.OPENROUTER_API_KEY),
    time: new Date().toISOString()
  });
}
