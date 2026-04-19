/* ============================================
   LOWTECH LANDING — MAIN JS
============================================ */

// Smart chatbot context (lazy-loaded; populated by initSmartContext)
const SmartContext = {
  ready: false,
  loading: false,
  error: null,
  search: null // async (queryText) => chunk | null
};

document.addEventListener('DOMContentLoaded', () => {
  initNavbar();
  initMobileMenu();
  initRevealOnScroll();
  initStatsCounter();
  initActiveNav();
  initPricingToggle();
  initChatbot();
  initFaq();
  initConversationDemo();
  // Fire-and-forget: embeddings load in background; Fuse serves until ready.
  initSmartContext().catch((e) => {
    console.warn('[Lowtech bot] smart context failed:', e);
    SmartContext.error = e;
  });
});

/* --------------------------------------------
   CONVERSATION DEMO — scroll-linked reveal
   Mapea el progreso de scroll dentro de la
   sección (0→1) a la visibilidad de cada
   mensaje y typing indicator, con fade fluido.
-------------------------------------------- */
function initConversationDemo() {
  const section = document.querySelector('.conv-demo');
  if (!section) return;

  const stage = section.querySelector('.conv-stage');
  const chatArea = section.querySelector('.wa-chat-area');
  const messages = Array.from(section.querySelectorAll('[data-reveal]'));
  const typings = Array.from(section.querySelectorAll('[data-typing-from]'));

  if (!stage || !chatArea || messages.length === 0) return;

  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (reduceMotion) {
    messages.forEach((m) => {
      m.style.opacity = 1;
      m.style.transform = 'none';
    });
    return;
  }

  // Each message is revealed over a small scroll window so it fades in
  // smoothly rather than popping. Adjust FADE for tighter/looser transitions.
  const FADE = 0.035;

  const clamp = (v, min, max) => Math.max(min, Math.min(max, v));
  const smooth = (t) => t * t * (3 - 2 * t); // smoothstep for nicer curve

  let ticking = false;

  const render = () => {
    const rect = stage.getBoundingClientRect();
    const scrollable = Math.max(1, rect.height - window.innerHeight);
    const progress = clamp(-rect.top / scrollable, 0, 1);

    // 1. Messages: cada uno colapsa a max-height 0 si aún no está revelado,
    //    así no ocupa espacio y no empuja el scroll innecesariamente.
    //    Una vez su scroll-progress pasa el umbral, expande a 150px y hace
    //    fade-in con opacity + translateY.
    messages.forEach((el) => {
      const start = parseFloat(el.dataset.reveal);
      if (progress < start) {
        el.style.opacity = '0';
        el.style.transform = 'translateY(8px)';
        el.style.maxHeight = '0px';
        return;
      }
      const p = smooth(clamp((progress - start) / FADE, 0, 1));
      el.style.opacity = String(p);
      el.style.transform = `translateY(${(1 - p) * 8}px)`;
      el.style.maxHeight = '150px';
    });

    // 2. Typing indicators: visible only inside their [from, to] window.
    //    Toggle class (not style) so CSS can collapse max-height and avoid
    //    reserving space when hidden.
    typings.forEach((el) => {
      const from = parseFloat(el.dataset.typingFrom);
      const to = parseFloat(el.dataset.typingTo);
      const visible = progress >= from && progress < to;
      el.classList.toggle('typing-on', visible);
    });

    // 3. Auto-scroll SOLO cuando el contenido excede el área visible. Mientras
    //    quepan todos los mensajes, el scroll se queda en 0 y el HOY se ve
    //    natural al comienzo del chat.
    if (rect.top < window.innerHeight && rect.bottom > 0) {
      if (chatArea.scrollHeight > chatArea.clientHeight + 2) {
        chatArea.scrollTop = chatArea.scrollHeight;
      } else {
        chatArea.scrollTop = 0;
      }
    }

    ticking = false;
  };

  const onScroll = () => {
    if (!ticking) {
      window.requestAnimationFrame(render);
      ticking = true;
    }
  };

  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll, { passive: true });
  render();
}

/* --------------------------------------------
   SMART CONTEXT — Transformers.js + lowtech.md
   Loads the MiniLM embedder and indexes every
   H2 section of content/lowtech.md so the bot
   can retrieve the most relevant block by
   semantic similarity (cosine dot product on
   normalized vectors).
-------------------------------------------- */
async function initSmartContext() {
  if (SmartContext.loading || SmartContext.ready) return;
  SmartContext.loading = true;

  // 1. Dynamic-import Transformers.js as an ES module.
  const mod = await import(
    'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2'
  );
  const { pipeline, env } = mod;
  env.allowLocalModels = false;
  env.useBrowserCache = true;

  // 2. Load a tiny quantized embedding model (~23MB, cached after first visit).
  const embedder = await pipeline(
    'feature-extraction',
    'Xenova/all-MiniLM-L6-v2',
    { quantized: true }
  );

  // 3. Fetch and parse the knowledge base. Always revalidate so edits to
  //    content/lowtech.md show up on the next page load without being
  //    masked by a stale browser cache.
  const res = await fetch('content/lowtech.md?v=' + Date.now(), { cache: 'no-cache' });
  if (!res.ok) throw new Error('No se pudo cargar content/lowtech.md');
  const md = await res.text();
  const chunks = parseMarkdownChunks(md);

  // 4. Embed each chunk's INTENT (match phrases), not its body. This way
  //    "qué es el chatbot" matches the chatbot chunk instead of the team
  //    chunk just because both sections start with "Somos un…".
  for (const c of chunks) {
    const out = await embedder(c.matchText, { pooling: 'mean', normalize: true });
    c.embedding = Array.from(out.data);
  }

  const cosine = (a, b) => {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  };

  SmartContext.search = async (query) => {
    const q = (query || '').trim();
    if (!q) return null;
    const out = await embedder(q, { pooling: 'mean', normalize: true });
    const qvec = Array.from(out.data);
    let best = { score: -Infinity, chunk: null };
    for (const c of chunks) {
      const s = cosine(qvec, c.embedding);
      if (s > best.score) best = { score: s, chunk: c };
    }
    // 0.55 is a decent empirical threshold for MiniLM on short queries.
    return best.score > 0.55 ? best.chunk : null;
  };

  SmartContext.ready = true;
  SmartContext.loading = false;
  console.info('[Lowtech bot] Smart context ready —', chunks.length, 'sections indexed.');
}

function parseMarkdownChunks(md) {
  const lines = md.split('\n');
  const out = [];
  let current = null;
  for (const line of lines) {
    const h2 = line.match(/^##\s+(.+?)\s*$/);
    if (h2) {
      if (current) out.push(current);
      current = { title: h2[1], content: '' };
    } else if (current) {
      current.content += line + '\n';
    }
  }
  if (current) out.push(current);

  // Extract the `<!-- match: a | b | c -->` comment per chunk — it's what
  // we use to embed intent. If absent, fall back to title + first 180 chars
  // of prose (stripped).
  for (const c of out) {
    const m = c.content.match(/<!--\s*match:\s*([^]+?)\s*-->/i);
    if (m) {
      const phrases = m[1].split('|').map((s) => s.trim()).filter(Boolean);
      c.matchText = c.title + '. ' + phrases.join('. ');
    } else {
      c.matchText = (c.title + '. ' + stripMarkdown(c.content)).slice(0, 240);
    }
  }
  return out;
}

function stripMarkdown(s) {
  return s
    .replace(/<!--[\s\S]*?-->/g, '')
    .replace(/[*_`]/g, '')
    .replace(/\[(.+?)\]\((.+?)\)/g, '$1')
    .replace(/^#+\s+/gm, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function chunkToAnswer(chunk) {
  const chipMatch = chunk.content.match(/<!--\s*chips:\s*(.+?)\s*-->/);
  const chips = chipMatch
    ? chipMatch[1].split('|').map((s) => s.trim()).filter(Boolean)
    : null;

  let body = chunk.content.replace(/<!--[\s\S]*?-->/g, '').trim();
  // Light markdown → chat-friendly plaintext.
  body = body.replace(/\*\*(.+?)\*\*/g, '$1');
  body = body.replace(/\*(.+?)\*/g, '$1');
  body = body.replace(/`(.+?)`/g, '$1');
  body = body.replace(/^[-*]\s+/gm, '• ');

  const paragraphs = body.split(/\n\s*\n/).map((s) => s.trim()).filter(Boolean);
  return {
    messages: paragraphs.length ? paragraphs : [body],
    chips: chips && chips.length ? chips : undefined
  };
}

/* --------------------------------------------
   NAVBAR SHADOW ON SCROLL
-------------------------------------------- */
function initNavbar() {
  const navbar = document.querySelector('.navbar');
  if (!navbar) return;

  const onScroll = () => {
    if (window.scrollY > 10) {
      navbar.classList.add('scrolled');
    } else {
      navbar.classList.remove('scrolled');
    }
  };

  onScroll();
  window.addEventListener('scroll', onScroll, { passive: true });
}

/* --------------------------------------------
   MOBILE MENU TOGGLE
-------------------------------------------- */
function initMobileMenu() {
  const toggle = document.querySelector('.nav-toggle');
  const menu = document.querySelector('.nav-mobile');
  if (!toggle || !menu) return;

  toggle.addEventListener('click', () => {
    toggle.classList.toggle('active');
    menu.classList.toggle('open');
  });

  menu.querySelectorAll('a').forEach((link) => {
    link.addEventListener('click', () => {
      toggle.classList.remove('active');
      menu.classList.remove('open');
    });
  });
}

/* --------------------------------------------
   REVEAL ON SCROLL (Intersection Observer)
   Staggered delay inside grids
-------------------------------------------- */
function initRevealOnScroll() {
  const targets = document.querySelectorAll('.reveal');
  if (!('IntersectionObserver' in window) || targets.length === 0) {
    targets.forEach((el) => el.classList.add('visible'));
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const el = entry.target;
          const delay = parseInt(el.dataset.delay || '0', 10);
          setTimeout(() => el.classList.add('visible'), delay);
          observer.unobserve(el);
        }
      });
    },
    { threshold: 0.12, rootMargin: '0px 0px -40px 0px' }
  );

  targets.forEach((el) => observer.observe(el));
}

/* --------------------------------------------
   STATS COUNTER ANIMATION
-------------------------------------------- */
function initStatsCounter() {
  const counters = document.querySelectorAll('[data-counter]');
  if (counters.length === 0) return;

  const animate = (el) => {
    const target = parseFloat(el.dataset.counter);
    const suffix = el.dataset.suffix || '';
    const prefix = el.dataset.prefix || '';
    const decimals = parseInt(el.dataset.decimals || '0', 10);
    const duration = 1600;
    const start = performance.now();

    const tick = (now) => {
      const elapsed = now - start;
      const t = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - t, 3);
      const current = target * eased;
      el.textContent = prefix + current.toFixed(decimals) + suffix;
      if (t < 1) requestAnimationFrame(tick);
      else el.textContent = prefix + target.toFixed(decimals) + suffix;
    };
    requestAnimationFrame(tick);
  };

  if (!('IntersectionObserver' in window)) {
    counters.forEach(animate);
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          animate(entry.target);
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.4 }
  );

  counters.forEach((el) => observer.observe(el));
}

/* --------------------------------------------
   ACTIVE NAV LINK (scrollspy — rAF, snappy)
   Picks the last section whose top is above the
   viewport anchor line (nav height + buffer).
-------------------------------------------- */
function initActiveNav() {
  const links = Array.from(document.querySelectorAll('.nav-links a[href^="#"]'));
  if (links.length === 0) return;

  const sections = links
    .map((l) => document.getElementById(l.getAttribute('href').slice(1)))
    .filter(Boolean);

  if (sections.length === 0) return;

  const ANCHOR_OFFSET = 100; // nav height (72) + breathing room
  let currentId = null;
  let ticking = false;

  const update = () => {
    const y = window.scrollY + ANCHOR_OFFSET;

    let active = sections[0];
    for (const s of sections) {
      if (s.offsetTop <= y) active = s;
      else break;
    }

    // Near bottom of page: force the last section (anchor line never reaches it)
    if (window.innerHeight + window.scrollY >= document.body.offsetHeight - 4) {
      active = sections[sections.length - 1];
    }

    if (active.id !== currentId) {
      currentId = active.id;
      const targetHref = '#' + active.id;
      links.forEach((link) => {
        link.classList.toggle('active', link.getAttribute('href') === targetHref);
      });
    }
    ticking = false;
  };

  const onScroll = () => {
    if (!ticking) {
      window.requestAnimationFrame(update);
      ticking = true;
    }
  };

  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll, { passive: true });
  update(); // initial highlight on load
}

/* --------------------------------------------
   PRICING BILLING TOGGLE
-------------------------------------------- */
function initPricingToggle() {
  const monthlyBtn = document.querySelector('[data-billing="monthly"]');
  const yearlyBtn = document.querySelector('[data-billing="yearly"]');
  const amounts = document.querySelectorAll('[data-price]');
  const periods = document.querySelectorAll('[data-period]');
  if (!monthlyBtn || !yearlyBtn) return;

  const setBilling = (mode) => {
    monthlyBtn.classList.toggle('active', mode === 'monthly');
    yearlyBtn.classList.toggle('active', mode === 'yearly');

    amounts.forEach((el) => {
      const base = parseFloat(el.dataset.price);
      const value = mode === 'yearly' ? Math.round(base * 0.8) : base;
      el.textContent = '$' + value;
    });

    periods.forEach((el) => {
      el.textContent = mode === 'yearly' ? '/mes · facturado anual' : '/mes';
    });
  };

  monthlyBtn.addEventListener('click', () => setBilling('monthly'));
  yearlyBtn.addEventListener('click', () => setBilling('yearly'));
}

/* --------------------------------------------
   CHATBOT WIDGET
-------------------------------------------- */
function initChatbot() {
  const container = document.getElementById('chat-messages');
  const input = document.getElementById('chat-input');
  const sendBtn = document.getElementById('chat-send');
  if (!container || !input || !sendBtn) return;

  const MAX_MESSAGES = 20;
  let messageCount = 0;
  let isBotTyping = false;

  // Knowledge base — each entry is a searchable FAQ about Lowtech.
  // `keywords` are matched fuzzily with Fuse.js; `q` is the canonical question shown in chips.
  const KB = [
    {
      id: 'que-es',
      keywords: 'que es chatbot definicion como funciona producto servicio hace lowtech asistente virtual bot ia inteligencia artificial',
      q: '¿Qué es el chatbot?',
      answer: {
        messages: [
          'Nuestro chatbot es un asistente con IA que atiende a tus clientes en WhatsApp las 24 horas 🤖',
          'Responde preguntas, toma pedidos, agenda citas — sin que estés pendiente.'
        ]
      }
    },
    {
      id: 'precio',
      keywords: 'precio costo cuanto cuesta plan planes tarifa caro barato vale mensual anual suscripcion pagar pago pagos factura facturacion presupuesto cotizar cotizacion',
      q: '¿Cuánto cuesta?',
      answer: {
        messages: [
          'Cada proyecto lo cotizamos a medida según lo que necesites 💼',
          'Ojo, este chat es solo una demo. Para una propuesta concreta, baja a la sección de contacto al final de la página.'
        ],
        chips: ['Ir a contacto ↓']
      }
    },
    {
      id: 'como-funciona',
      keywords: 'como funciona pasos proceso flujo implementacion implementar integrar setup configurar',
      q: '¿Cómo funciona?',
      answer: {
        messages: [
          'Es muy sencillo ⚡ En 3 pasos:',
          '1️⃣ Nos cuentas tu negocio\n2️⃣ Configuramos tu chatbot\n3️⃣ ¡Empieza a atender solo!'
        ]
      }
    },
    {
      id: 'tiempo',
      keywords: 'cuanto tarda tiempo demora duracion semanas dias rapido urgente cuando listo',
      q: '¿Cuánto tarda la implementación?',
      answer: {
        messages: [
          'Tu chatbot está en producción entre 5 y 10 días hábiles ⏱️',
          'Depende de cuánta info tengas lista (catálogo, precios, FAQs) y de las integraciones.'
        ]
      }
    },
    {
      id: 'industrias',
      keywords: 'industria industrias sectores rubro negocio para quien sirve funciona hotel hoteles clinica salud medicina inmobiliaria inmobiliaria restaurante tienda comercio',
      q: '¿Para qué industrias sirve?',
      answer: {
        messages: [
          'Funciona para cualquier industria 🎯',
          'Hemos implementado en hoteles, clínicas, inmobiliarias, tiendas y servicios. Si tu negocio atiende por WhatsApp, sirve.'
        ]
      }
    },
    {
      id: 'tecnico',
      keywords: 'necesito conocimiento tecnico programacion codigo dificil saber dominar aprender curso',
      q: '¿Necesito conocimientos técnicos?',
      answer: {
        messages: [
          'Cero conocimientos técnicos requeridos 🙌',
          'Nosotros hacemos todo el setup. Tú solo recibes un panel simple para ver conversaciones y editar respuestas sin código.'
        ]
      }
    },
    {
      id: 'no-sabe',
      keywords: 'no sabe no responde no conoce pregunta dificil extrana rara humano derivar escalar',
      q: '¿Y si no sabe responder?',
      answer: {
        messages: [
          'El bot deriva la conversación a un humano de tu equipo y te avisa 🙋',
          'Además registra la pregunta para que la próxima vez ya sepa responderla.'
        ]
      }
    },
    {
      id: 'cancelar',
      keywords: 'cancelar cancelacion permanencia contrato salir darse de baja terminar finalizar deshacer devolver dinero',
      q: '¿Puedo cancelar cuando quiera?',
      answer: {
        messages: [
          'Sí, sin permanencia ✅ Trabajamos mes a mes, sin contratos largos.',
          'Si decides irte, exportamos tus datos y cerramos la cuenta el mismo día.'
        ]
      }
    },
    {
      id: 'internacional',
      keywords: 'internacional fuera venezuela colombia mexico panama espana paises exterior otro pais extranjero regional',
      q: '¿Trabajan fuera de Venezuela?',
      answer: {
        messages: [
          'Por ahora trabajamos solo en Venezuela 🇻🇪',
          'Queremos consolidar bien nuestra operación local antes de expandirnos. Si estás fuera y te interesa lo que hacemos, igual escríbenos — te avisamos apenas abramos otros mercados.'
        ],
        chips: ['Ir a contacto ↓']
      }
    },
    {
      id: 'otros-servicios',
      keywords: 'apps aplicaciones movil mobile web pagina website desarrollo software automatizacion zapier make n8n',
      q: '¿Hacen apps y webs también?',
      answer: {
        messages: [
          'Sí. Además de chatbots hacemos páginas web, apps móviles y automatizaciones a medida 🛠️'
        ],
        chips: ['Ver casos de uso', 'Ir a contacto ↓']
      }
    },
    {
      id: 'demo',
      keywords: 'demo demostracion probar prueba gratis ejemplo quiero contactar contacto asesor hablar llamar agendar cita reunion',
      q: 'Quiero contactarlos',
      answer: {
        messages: [
          'Recuerda que este es un chat de demo, no puedo tomar tus datos desde aquí.',
          'Para que te contactemos, baja a la sección de contacto al final de la página — ahí tienes el formulario y el WhatsApp directo.'
        ],
        chips: ['Ir a contacto ↓']
      }
    },
    {
      id: 'equipo',
      keywords: 'equipo gente personas fundador ceo quien somos nosotros empresa acerca sobre about',
      q: '¿Quiénes son ustedes?',
      answer: {
        messages: [
          'Somos Lowtech, un equipo pequeño de venezolanos 🇻🇪',
          'Puedes ver al equipo en la sección "Nuestro equipo" más arriba.'
        ]
      }
    },
    {
      id: 'whatsapp',
      keywords: 'whatsapp wa numero telefono integracion business api oficial meta',
      q: '¿Cómo se integra con WhatsApp?',
      answer: {
        messages: [
          'Usamos la WhatsApp Business API oficial de Meta 📲',
          'Nosotros hacemos todo el proceso de verificación. Tú solo nos das el número que quieras usar.'
        ]
      }
    },
    {
      id: 'saludo',
      keywords: 'hola hey buenas saludos buen dia tardes noches hi hello',
      q: 'Hola',
      answer: {
        messages: [
          '¡Hola! 👋 Encantado de saludarte.',
          '¿Sobre qué de Lowtech te gustaría saber?'
        ],
        chips: ['¿Qué es el chatbot?', '¿Cómo funciona?', '¿Para qué industrias?', 'Ir a contacto ↓']
      }
    },
    {
      id: 'gracias',
      keywords: 'gracias thanks genial chevere perfecto buenisimo excelente ok bien vale',
      q: 'Gracias',
      answer: {
        messages: [
          '¡Con gusto! 🙌',
          'Recuerda: para hablar con el equipo humano, baja a la sección de contacto al final de la página.'
        ],
        chips: ['Ir a contacto ↓']
      }
    }
  ];

  const fallback = {
    messages: [
      'No estoy seguro de cómo responder a eso 🤔',
      'Si quieres, pregúntame sobre Lowtech, o baja a la sección de contacto al final de la página para escribirle al equipo directo.'
    ],
    chips: ['Ir a contacto ↓']
  };

  // SAFETY: never handle personal/sensitive info, credentials, or prompt-injection.
  const SENSITIVE_PATTERNS = [
    /\bc[eé]dul/i,
    /\bdni\b/i,
    /\bpasaport/i,
    /\brif\b/i,
    /\bcurp\b/i,
    /\bssn\b/i,
    /tarjeta.*(cr[eé]dito|d[eé]bito|bancaria)/i,
    /cuenta.*(bancari|corriente|ahorro)/i,
    /\bcvv\b|\bcvc\b|\biban\b|\bswift\b/i,
    /(contrase[ñn]a|\bpassword\b|\bclave\b|\bpin\b)/i,
    /(api[- _]?key|\btoken\b|credencial|acceso.*(cuenta|sistema))/i,
    /(dato|informaci[oó]n).?(personal|privad|confidencial|sensib)/i,
    /(email|correo|tel[eé]fono|celular|whatsapp).?(personal|privado|directo|del (ceo|cto|fundador|equipo))/i,
    /direcci[oó]n.*(casa|domicil|hogar|residen)/i,
    /d[oó]nde.?viv/i,
    /(ignor|olvida).?(las )?(instruc|prompt|reglas|sistem)/i,
    /(system.?prompt|jailbreak|dan.?mode|act[uú]a como|finge ser|haz de cuenta)/i,
    /dame.*(\bemail\b|\btel[eé]fono\b|\bn[uú]mero\b).*(emmanuel|andr[eé]s|valentina|daniela)/i
  ];

  const securityDeflection = {
    messages: [
      'Por seguridad no puedo compartir información personal ni datos sensibles 🔒',
      'Si lo necesitas, baja a la sección de contacto al final de la página y escríbele directo al equipo humano.'
    ],
    chips: ['Ir a contacto ↓']
  };

  const isSensitive = (t) => SENSITIVE_PATTERNS.some((p) => p.test(t));

  // Fuse.js fuzzy search over keywords + canonical question
  const fuse = (typeof Fuse !== 'undefined')
    ? new Fuse(KB, {
        keys: [
          { name: 'keywords', weight: 0.7 },
          { name: 'q', weight: 0.3 }
        ],
        threshold: 0.4,
        ignoreLocation: true,
        minMatchCharLength: 3,
        includeScore: true
      })
    : null;

  // LLM proxy endpoint:
  //  - En dev con live-server (localhost:3000), el proxy corre en un puerto distinto (:3001).
  //  - En prod (Vercel) el mismo dominio sirve tanto el estático como la función serverless,
  //    así que usamos una URL relativa.
  const isLocalDev =
    (location.hostname === 'localhost' || location.hostname === '127.0.0.1') &&
    location.port !== '3001';
  const LLM_API = isLocalDev ? 'http://localhost:3001/api/chat' : '/api/chat';

  // Conversation history (OpenAI/Groq schema): [{ role: 'user'|'assistant', content }].
  const chatHistory = [];
  const HISTORY_MAX_TURNS = 6; // keep last 6 pairs (12 messages)

  const askLLM = async (message) => {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 20000);
    try {
      const res = await fetch(LLM_API, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, history: chatHistory }),
        signal: controller.signal
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        const err = new Error(data.error || 'HTTP ' + res.status);
        err.status = res.status;
        throw err;
      }
      const data = await res.json();
      if (!data.answer) throw new Error('Respuesta vacía');

      // Track turn in history (OpenAI/Groq schema)
      chatHistory.push({ role: 'user', content: message });
      chatHistory.push({ role: 'assistant', content: data.answer });
      while (chatHistory.length > HISTORY_MAX_TURNS * 2) chatHistory.shift();

      // Split the answer into paragraphs so the bot sends them as separate
      // messages with typing indicators between — feels more human.
      const paragraphs = data.answer
        .split(/\n\s*\n|(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ¡¿])/)
        .map((s) => s.trim())
        .filter(Boolean);
      const messages = paragraphs.length >= 1 && paragraphs.length <= 4
        ? paragraphs
        : [data.answer.trim()];

      return { id: 'llm', answer: { messages } };
    } finally {
      clearTimeout(timeout);
    }
  };

  const classify = async (raw) => {
    const rawLower = raw.toLowerCase().trim();
    const t = rawLower.normalize('NFD').replace(/[\u0300-\u036f]/g, '');

    if (!t) return null;

    // 1. Security gate — always first, before any API call.
    if (isSensitive(rawLower) || isSensitive(t)) {
      return { id: '__sensitive', answer: securityDeflection };
    }

    // 2. Pricing gate — always redirect, never give numbers. Saves an API call
    //    and guarantees the behavior even if the model hallucinates.
    if (/(cuesta|precio|plan|tarifa|cotizac|presupuesto|cuanto|cu[aá]nto vale)/.test(t)) {
      return KB.find((x) => x.id === 'precio');
    }

    // 3. Primary: LLM (Groq) via our proxy — generative, context-aware.
    try {
      return await askLLM(raw);
    } catch (err) {
      console.warn('[Lowtech bot] LLM unavailable, falling back:', err.message);
    }

    // 4. Fallback: local semantic retrieval from content/lowtech.md.
    if (SmartContext.ready && SmartContext.search) {
      try {
        const chunk = await SmartContext.search(t);
        if (chunk) {
          return { id: 'smart:' + chunk.title, answer: chunkToAnswer(chunk) };
        }
      } catch (err) {
        console.warn('[Lowtech bot] semantic search error:', err);
      }
    }

    // 5. Fuse keyword fallback.
    if (fuse) {
      const results = fuse.search(t);
      if (results.length > 0 && results[0].score < 0.45) {
        return results[0].item;
      }
    }

    // 6. Regex last resort.
    if (/(que es|chatbot|que hace)/.test(t)) return KB.find((x) => x.id === 'que-es');
    if (/(como funciona|pasos|proceso)/.test(t)) return KB.find((x) => x.id === 'como-funciona');
    if (/(demo|contact|asesor|hablar)/.test(t)) return KB.find((x) => x.id === 'demo');

    return null;
  };

  const timestamp = () => {
    const d = new Date();
    const hh = String(d.getHours()).padStart(2, '0');
    const mm = String(d.getMinutes()).padStart(2, '0');
    return `${hh}:${mm}`;
  };

  const scrollBottom = () => {
    container.scrollTop = container.scrollHeight;
  };

  const maybeReset = () => {
    if (messageCount >= MAX_MESSAGES) {
      container.innerHTML = '';
      const sys = document.createElement('div');
      sys.className = 'msg system';
      sys.textContent = '— Chat reiniciado —';
      container.appendChild(sys);
      messageCount = 0;
    }
  };

  const removeChips = () => {
    container.querySelectorAll('.chips').forEach((c) => c.remove());
  };

  const addMessage = (text, sender) => {
    maybeReset();
    const msg = document.createElement('div');
    msg.className = `msg ${sender}`;
    const body = document.createElement('span');
    body.textContent = text;
    const time = document.createElement('span');
    time.className = 'msg-time';
    time.textContent = timestamp();
    msg.appendChild(body);
    msg.appendChild(time);
    container.appendChild(msg);
    messageCount++;
    scrollBottom();
    return msg;
  };

  // Chips de acción (scroll). Mapa explícito de "etiqueta normalizada" → id de sección.
  // Cualquier chip cuyo label caiga acá se convierte en un navegador visual en vez
  // de mandarlo como pregunta al LLM — así no depende de que el servidor responda.
  const CHIP_SCROLL_TARGETS = {
    'ir a contacto': 'final-cta',
    'ir al contacto': 'final-cta',
    'bajar a contacto': 'final-cta',
    'ver la seccion de contacto': 'final-cta',
    'hablar con un asesor': 'final-cta',
    'dejar mis datos': 'final-cta',
    'hablar por whatsapp': 'final-cta',
    'ver casos de uso': 'casos',
    'ver casos': 'casos',
    'casos de uso': 'casos',
    'ver que hacen': 'chatbot',
    'ver que si hacen': 'chatbot',
    'ver que sí hacen': 'chatbot',
    'probar el chatbot': 'demo',
    'ver precios': 'final-cta',
    'ver equipo': 'equipo',
    'ver el equipo': 'equipo',
    'ver preguntas frecuentes': 'faq',
    'ver faq': 'faq'
  };

  const normalizeChip = (s) =>
    s
      .toLowerCase()
      .normalize('NFD')
      .replace(/[\u0300-\u036f]/g, '')
      .replace(/[↓→¿?¡!.,]/g, '')
      .replace(/\s+/g, ' ')
      .trim();

  const getScrollTarget = (label) => CHIP_SCROLL_TARGETS[normalizeChip(label)] || null;

  const scrollToSection = (id) => {
    const target = document.getElementById(id);
    if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  const addChips = (labels) => {
    if (!labels || labels.length === 0) return;
    removeChips();
    const wrap = document.createElement('div');
    wrap.className = 'chips';
    labels.forEach((label) => {
      const chip = document.createElement('button');
      chip.className = 'chip';
      chip.type = 'button';
      chip.textContent = label;
      const scrollId = getScrollTarget(label);
      if (scrollId) chip.classList.add('chip-action');
      chip.addEventListener('click', () => {
        if (isBotTyping) return;
        if (scrollId) {
          scrollToSection(scrollId);
          return;
        }
        handleUserInput(label);
      });
      wrap.appendChild(chip);
    });
    container.appendChild(wrap);
    scrollBottom();
  };

  const showTyping = () => {
    const typing = document.createElement('div');
    typing.className = 'typing';
    typing.id = 'typing-indicator';
    typing.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';
    container.appendChild(typing);
    scrollBottom();
  };

  const hideTyping = () => {
    const el = document.getElementById('typing-indicator');
    if (el) el.remove();
  };

  const delay = (min = 800, max = 1200) =>
    new Promise((res) => setTimeout(res, Math.random() * (max - min) + min));

  // Debug mode: if URL has ?debug=1, every bot message shows a small badge
  // with the layer that produced it (llm | smart | fuse | regex | sensitive).
  const DEBUG = new URLSearchParams(location.search).get('debug') === '1';

  const sourceLabel = (id) => {
    if (!id) return { label: 'fallback', kind: 'fallback' };
    if (id === '__sensitive') return { label: 'seguridad (regex)', kind: 'security' };
    if (id === 'llm') return { label: 'LLM (OpenRouter)', kind: 'llm' };
    if (id.startsWith('smart:')) return { label: 'embeddings locales', kind: 'smart' };
    if (id.startsWith('regex:')) return { label: 'regex', kind: 'regex' };
    // KB hits use their plain id (precio, demo, etc.)
    return { label: 'Fuse (KB local)', kind: 'fuse' };
  };

  const sendBotSequence = async (entry) => {
    isBotTyping = true;
    sendBtn.disabled = true;
    removeChips();

    const payload = entry && entry.answer ? entry.answer : entry || fallback;
    const id = (entry && entry.id) || 'fallback';
    const src = sourceLabel(id);

    // Always log to DevTools so you can inspect prod behavior without toggling UI.
    console.info(
      `%c[Lowtech bot]%c answered via ${src.label} (id="${id}")`,
      'color:#185FA5;font-weight:700',
      'color:inherit'
    );

    for (const text of payload.messages) {
      showTyping();
      await delay();
      hideTyping();
      const msgEl = addMessage(text, 'bot');
      if (DEBUG && msgEl) {
        const badge = document.createElement('span');
        badge.className = 'msg-source src-' + src.kind;
        badge.textContent = 'vía ' + src.label;
        msgEl.appendChild(badge);
      }
    }

    if (payload.chips && payload.chips.length) {
      await delay(200, 400);
      addChips(payload.chips);
    }

    isBotTyping = false;
    sendBtn.disabled = false;
    input.focus();
  };

  const handleUserInput = async (text) => {
    const clean = text.trim();
    if (!clean || isBotTyping) return;
    addMessage(clean, 'user');
    input.value = '';
    try {
      const hit = await classify(clean);
      sendBotSequence(hit || fallback);
    } catch (err) {
      console.warn('[Lowtech bot] classify error:', err);
      sendBotSequence(fallback);
    }
  };

  sendBtn.addEventListener('click', () => handleUserInput(input.value));
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleUserInput(input.value);
    }
  });

  // Initial greeting
  (async () => {
    isBotTyping = true;
    sendBtn.disabled = true;
    await delay(400, 700);

    showTyping();
    await delay();
    hideTyping();
    addMessage('¡Hola! 👋 Soy el asistente virtual de Lowtech — esto es una demo de cómo atiende nuestro bot.', 'bot');

    showTyping();
    await delay();
    hideTyping();
    addMessage('Puedo resolverte dudas sobre lo que hacemos. Para contactar al equipo humano, usa la sección de contacto al final de la página.', 'bot');

    await delay(200, 400);
    addChips([
      '¿Qué es el chatbot?',
      '¿Cómo funciona?',
      '¿Para qué industrias?',
      'Ir a contacto ↓'
    ]);

    isBotTyping = false;
    sendBtn.disabled = false;
  })();
}

/* --------------------------------------------
   FAQ ACCORDION — close others when one opens
-------------------------------------------- */
function initFaq() {
  const items = document.querySelectorAll('.faq-item');
  if (items.length === 0) return;

  items.forEach((item) => {
    item.addEventListener('toggle', () => {
      if (item.open) {
        items.forEach((other) => {
          if (other !== item) other.open = false;
        });
      }
    });
  });
}
