
function translateChainlitInterface() {
  
  const observer = new MutationObserver((mutations) => {
    
    const dialogTitle = document.querySelector('.cl-modal-title, [role="dialog"] h2, [role="dialog"] .text-lg');
    if (dialogTitle && dialogTitle.textContent.includes('Create New Chat')) {
      dialogTitle.textContent = 'Criar Novo Chat';
    }
    
    
    const confirmText = document.querySelector('[role="dialog"] p, .cl-modal-content p');
    if (confirmText && confirmText.textContent.includes('This will clear your current chat history')) {
      confirmText.textContent = 'Isso irá limpar seu histórico de conversa atual. Tem certeza que deseja continuar?';
    }
    
    
    const buttons = document.querySelectorAll('[role="dialog"] button');
    buttons.forEach(button => {
      if (button.textContent.trim() === 'Cancel') {
        button.textContent = 'Cancelar';
      }
      if (button.textContent.trim() === 'Continue' || button.textContent.trim() === 'Confirm') {
        button.textContent = 'Continuar';
      }
    });
  });
  
  
  observer.observe(document.body, { 
    childList: true, 
    subtree: true,
    characterData: true
  });
  
  
  const translateNewChatButton = () => {
    const newChatButton = document.querySelector('button[aria-label="New Chat"], a[href="/"], .cl-new-chat');
    if (newChatButton) {
      
      const textNode = Array.from(newChatButton.childNodes).find(node => 
        node.nodeType === Node.TEXT_NODE && node.textContent.trim().length > 0
      );
      
      if (textNode && textNode.textContent.includes('New Chat')) {
        textNode.textContent = 'Novo Chat';
      }
    }
  };
  
  
  translateNewChatButton();
  setInterval(translateNewChatButton, 2000);
}


document.addEventListener('DOMContentLoaded', function() {
  replaceFavicon();
  translateChainlitInterface();

    
  function getCurrentTheme() {
    
    const dataTheme = document.documentElement.getAttribute('data-theme');
    if (dataTheme) return dataTheme;
    
    
    const storedTheme = localStorage.getItem('theme');
    if (storedTheme) return storedTheme;
    
    
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      return 'dark';
    }
    
    
    return 'dark';
  }
  
  function replaceLogo() {
    const theme = detectCurrentTheme();
    const logoPath = theme === 'dark' ? '/public/logo-dark2.png' : '/public/logo-light2.png';
    
    
    const avatarImages = document.querySelectorAll('img[alt="Avatar for Sistema Pericial"]');
    avatarImages.forEach(img => {
      img.src = logoPath;
      img.style.width = '20px';
      img.style.height = '20px';
      img.style.objectFit = 'contain';
    });
  }
  
  function replaceFavicon() {
    const theme = detectCurrentTheme();
    const faviconPath = theme === 'dark' ? '/public/logo-light2.png' : '/public/logo-dark2.png';
    
    
    const existingFavicons = document.querySelectorAll('link[rel="icon"], link[rel="shortcut icon"]');
    existingFavicons.forEach(favicon => favicon.remove());
    
    
    const newFavicon = document.createElement('link');
    newFavicon.rel = 'icon';
    newFavicon.href = faviconPath;
    newFavicon.type = 'image/x-icon';
    document.head.appendChild(newFavicon);
  }

  
  function updateThemeElements() {
    replaceLogo();
    replaceFavicon();
  }

  
  updateThemeElements();
  
  
  const observer = new MutationObserver(function(mutations) {
    updateThemeElements();
  });
  
  
  observer.observe(document.body, { 
    childList: true, 
    subtree: true,
    attributes: true,
    attributeFilter: ['data-theme']
  });
  
  
  const themeObserver = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
      if (mutation.attributeName === 'data-theme') {
        updateThemeElements();
      }
    });
  
  
  themeObserver.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['data-theme']
  });
  
  
  window.addEventListener('storage', function(event) {
    if (event.key === 'theme') {
      updateThemeElements();
    }
  });
});


function updateThemeAssets() {
  
  const isDarkTheme = document.documentElement.getAttribute('data-theme') === 'dark';
  
  
  const logoPath = isDarkTheme ? '/public/logo-dark2.png' : '/public/logo-light2.png';
  const faviconPath = isDarkTheme ? '/public/logo-dark2.png' : '/public/logo-light2.png';
  
  console.log('Tema atual:', isDarkTheme ? 'dark' : 'light');
  console.log('Usando logo:', logoPath);
  
  
  const avatarImages = document.querySelectorAll('img[alt="Avatar for Sistema Pericial"]');
  avatarImages.forEach(img => {
    img.src = logoPath;
    console.log('Logo substituída:', img);
  });
  
  
  let favicon = document.querySelector('link[rel="icon"]');
  if (!favicon) {
    favicon = document.createElement('link');
    favicon.rel = 'icon';
    favicon.type = 'image/x-icon';
    document.head.appendChild(favicon);
  }
  favicon.href = faviconPath;
}


function setupThemeObserver() {
  
  const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      if (mutation.attributeName === 'data-theme') {
        console.log('Tema alterado para:', document.documentElement.getAttribute('data-theme'));
        updateThemeAssets();
      }
    });
  });
  
  observer.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['data-theme']
  });
  
  console.log('Observador de tema configurado');
}

  
  function setupPeriodicCheck() {
    
    setInterval(() => {
      updateThemeAssets();
    }, 1000);
    
    console.log('Verificação periódica configurada');
  }

  
  document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM carregado, inicializando...');
    
    
    updateThemeAssets();
    
    
    setupThemeObserver();
    
    
    setupPeriodicCheck();
    
    
    document.addEventListener('click', (event) => {
      
      if (event.target.closest('button[aria-label="Toggle theme"]')) {
        console.log('Botão de tema clicado');
        
        setTimeout(updateThemeAssets, 100);
      }
    });
    
    console.log('Inicialização completa');
  });

  
  window.addEventListener('load', () => {
    console.log('Página totalmente carregada');
    updateThemeAssets();
  });


  function detectCurrentTheme() {
    
    const htmlEl = document.documentElement;
    const bodyEl = document.body;
    
    if (htmlEl.classList.contains('dark') || bodyEl.classList.contains('dark')) {
      return 'dark';
    } else if (htmlEl.classList.contains('light') || bodyEl.classList.contains('light')) {
      return 'light';
    }
    
    
    const dataTheme = htmlEl.getAttribute('data-theme') || bodyEl.getAttribute('data-theme');
    if (dataTheme) {
      return dataTheme;
    }
    
    
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      return 'dark';
    }
    
    
    return 'dark';
  }
  
  
  function getThemeColors() {
    const theme = detectCurrentTheme();
    
    if (theme === 'light') {
      return {
        background: '#ffffff',
        secondaryBackground: '#f5f5f5',
        accentColor: '#f5145f',
        textColor: '#333333',
        borderColor: '#e0e0e0',
        promptBackground: '#f5f5f5',
        promptButtonBg: '#f5145f',
        promptButtonText: '#ffffff',
        scrollbarTrack: '#f0f0f0',
        scrollbarThumb: '#c0c0c0'
      };
    } else {
      
      return {
        background: '#121212',
        secondaryBackground: '#1e1e1e',
        accentColor: '#f5145f',
        textColor: '#ffffff',
        borderColor: '#333333',
        promptBackground: '#1e1e1e',
        promptButtonBg: '#f5145f',
        promptButtonText: '#ffffff',
        scrollbarTrack: '#121212',
        scrollbarThumb: '#555555'
      };
    }
  }

function replaceFavicon() {
  
  const existingFavicons = document.querySelectorAll('link[rel*="icon"]');
  existingFavicons.forEach(favicon => favicon.remove());
  
  
  const favicon = document.createElement('link');
  favicon.rel = 'icon';
  favicon.href = '/logo.svg'; 
  favicon.type = 'image/svg+xml';
  document.head.appendChild(favicon);
  
  
  const observer = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
      if (mutation.addedNodes) {
        mutation.addedNodes.forEach(function(node) {
          if (node.nodeName === 'LINK' && 
              (node.rel === 'icon' || node.rel === 'shortcut icon') && 
              node.href.includes('chainlit')) {
            
            node.remove();
            document.head.appendChild(favicon.cloneNode(true));
          }
        });
      }
    });
  });
  
  
  observer.observe(document.head, { childList: true, subtree: true });
  
  
  setInterval(function() {
    const chainlitFavicons = document.querySelectorAll('link[rel*="icon"][href*="chainlit"]');
    if (chainlitFavicons.length > 0) {
      chainlitFavicons.forEach(favicon => favicon.remove());
      document.head.appendChild(favicon.cloneNode(true));
    }
  }, 1000);
}

  
  function applyThemeColors() {
    const colors = getThemeColors();
    
    
    const panel = document.getElementById('prompt-panel');
    if (panel) {
      panel.style.backgroundColor = colors.promptBackground;
      panel.style.borderColor = colors.borderColor;
      panel.style.color = colors.textColor;
      
      
      const title = panel.querySelector('h3');
      if (title) {
        title.style.color = colors.textColor;
      }
      
      
      const toggleButton = panel.querySelector('button');
      if (toggleButton) {
        toggleButton.style.color = colors.textColor;
      }
      
      
      const header = panel.querySelector('div:first-child');
      if (header) {
        header.style.borderBottomColor = colors.borderColor;
      }
      
      
      const buttons = panel.querySelectorAll('.prompt-button');
      buttons.forEach(button => {
        button.style.backgroundColor = colors.promptButtonBg;
        button.style.color = colors.promptButtonText;
      });
    }
  
  
  const style = document.getElementById('theme-variables') || document.createElement('style');
  if (!style.id) {
    style.id = 'theme-variables';
  }
  
  style.textContent = `
    :root {
      --primary-background: ${colors.background};
      --secondary-background: ${colors.secondaryBackground};
      --accent-color: ${colors.accentColor};
      --text-color: ${colors.textColor};
      --border-color: ${colors.borderColor};
    }
    
    ::-webkit-scrollbar-track {
      background: ${colors.scrollbarTrack};
    }
    
    ::-webkit-scrollbar-thumb {
      background: ${colors.scrollbarThumb};
    }
    
    ::-webkit-scrollbar-thumb:hover {
      background: ${colors.scrollbarThumb === '#555555' ? '#777777' : '#999999'};
    }
  `;
  
  if (!style.parentNode) {
    document.head.appendChild(style);
  }
}
  
  
  function createPromptPanel() {
    
    if (document.getElementById('prompt-panel')) {
      return;
    }
    
    const colors = getThemeColors();
    
    
    const prompts = [
      "NOME DA VARA",
      "NÚMERO DA AÇÃO",
      "NOME DO AUTOR DO PROCESSO",
      "NOME DO(S) RÉU(S)",
      "NOME DO MÉDICO PERITO",
      "DATA, HORA E LOCAL DA REALIZAÇÃO DA PERÍCIA MÉDICA",
      "NOME DA CIDADE DE PROCEDÊNCIA DA VARA, DATA DA REALIZAÇÃO DA PERÍCIA",
      "IDENTIFICAÇÃO DO RECLAMANTE",
      "ALEGAÇÃO DO/A RECLAMANTE",
      "IDENTIFICAÇÃO DA RECLAMADA",
      "ALEGAÇÃO DA RECLAMADA",
      "PARTICIPANTES DO ATO PERICIAL",
      "DATA EM QUE O/A RECLAMANTE FOI ADMITIDO",
      "DATA EM QUE O/A RECLAMANTE FOI DEMITIDO",
      "NOME DA FUNÇÃO QUE O RECLAMANTE DESEMPENHAVA NA RECLAMADA",
      "SETOR EM QUE O RECLAMANTE TRABALHA",
      "FUNÇÃO PARA QUAL O RECLAMANTE FOI CONTRATADO",
      "DIAS DE TRABALHO NA SEMANA OU TIPO DE ESCALA E HORÁRIO DE TRABALHO",
      "NÚMERO DE PAUSAS OU INTERVALOR E QUANTIDADE DE TEMPO",
      "EXAMES QUE FORAM REALIZADOS",
      "CURSOS QUE O RECLAMANTE FOI SUBMETIDO",
      "ANTECEDENTES PREVIDENCIÁRIOS",
      "TRAZER TODAS AS INFORMAÇÕES RELEVANTES PARA UM AGENTE PERICÍAL NESSE DOCUMENTO"
    ];
    
    
    const panel = document.createElement('div');
    panel.id = 'prompt-panel';
    panel.style.position = 'fixed';
    panel.style.left = '20px';
    panel.style.right = 'auto'; 
    panel.style.top = '80px';
    panel.style.width = '300px';
    panel.style.maxHeight = 'calc(100vh - 100px)';
    panel.style.overflowY = 'auto';
    panel.style.backgroundColor = colors.promptBackground;
    panel.style.border = `1px solid ${colors.borderColor}`;
    panel.style.borderRadius = '8px';
    panel.style.padding = '10px';
    panel.style.zIndex = '1000';
    panel.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.5)';
    panel.style.transition = 'none'; 
    panel.style.color = colors.textColor;
    
    
    const header = document.createElement('div');
    header.style.padding = '10px';
    header.style.borderBottom = `1px solid ${colors.borderColor}`;
    header.style.marginBottom = '10px';
    header.style.display = 'flex';
    header.style.justifyContent = 'space-between';
    header.style.alignItems = 'center';
    header.style.cursor = 'move'; 
    header.style.userSelect = 'none'; 
    
    const title = document.createElement('h3');
    title.textContent = 'Prompts Periciais';
    title.style.margin = '0';
    title.style.color = colors.textColor;
    title.style.fontSize = '16px';
    
    const toggleButton = document.createElement('button');
    toggleButton.innerHTML = '−'; 
    toggleButton.style.background = 'none';
    toggleButton.style.border = 'none';
    toggleButton.style.color = colors.textColor;
    toggleButton.style.fontSize = '20px';
    toggleButton.style.cursor = 'pointer';
    toggleButton.style.padding = '0 5px';
    toggleButton.title = 'Minimizar/Maximizar';
    
    header.appendChild(title);
    header.appendChild(toggleButton);
    panel.appendChild(header);
    
    
    const promptContainer = document.createElement('div');
    promptContainer.id = 'prompt-container';
    promptContainer.style.display = 'flex';
    promptContainer.style.flexDirection = 'column';
    promptContainer.style.gap = '8px';
    panel.appendChild(promptContainer);
    
    
    prompts.forEach(prompt => {
      const button = document.createElement('button');
      button.textContent = prompt;
      button.className = 'prompt-button';
      button.style.backgroundColor = colors.promptButtonBg;
      button.style.color = colors.promptButtonText;
      button.style.border = 'none';
      button.style.borderRadius = '4px';
      button.style.padding = '8px 12px';
      button.style.textAlign = 'left';
      button.style.cursor = 'pointer';
      button.style.transition = 'background-color 0.2s';
      button.style.fontSize = '14px';
      button.style.overflow = 'hidden';
      button.style.textOverflow = 'ellipsis';
      button.style.whiteSpace = 'nowrap';
      button.setAttribute('data-prompt', prompt); 

      button.addEventListener('mouseover', function() {
        this.style.filter = 'brightness(1.2)';
      });
      
      button.addEventListener('mouseout', function() {
        this.style.filter = 'brightness(1)';
      });
      
      button.addEventListener('click', function() {
        insertPrompt(prompt);
      });
      
      promptContainer.appendChild(button);
    });
    
    
    document.body.appendChild(panel);
    
    
    toggleButton.addEventListener('click', function(e) {
      e.stopPropagation(); 
      const container = document.getElementById('prompt-container');
      if (container.style.display === 'none') {
        container.style.display = 'flex';
        this.innerHTML = '−'; 
      } else {
        container.style.display = 'none';
        this.innerHTML = '+'; 
      }
    });
    
    
    makeDraggableFast(panel, header);
  }

  
  
  function insertPrompt(prompt) {
    console.log("Tentando inserir prompt:", prompt);
    
    
    const inputField = document.querySelector('.cl-chat-input, textarea[placeholder], div[contenteditable="true"]');
    
    if (!inputField) {
      console.error("Campo de entrada não encontrado!");
      return;
    }
    
    console.log("Campo de entrada encontrado:", inputField);
    
    
    const isContentEditable = inputField.getAttribute('contenteditable') === 'true';
    
    if (isContentEditable) {
      
      inputField.textContent = prompt;
      inputField.focus();
      
      
      inputField.dispatchEvent(new Event('input', { bubbles: true }));
      inputField.dispatchEvent(new Event('change', { bubbles: true }));
    } else {
      
      inputField.value = prompt;
      inputField.focus();
      
      
      inputField.dispatchEvent(new Event('input', { bubbles: true }));
      inputField.dispatchEvent(new Event('change', { bubbles: true }));
      
      
      for (const char of prompt.split('')) {
        const keyEvent = new KeyboardEvent('keydown', {
          key: char,
          bubbles: true,
          cancelable: true
        });
        inputField.dispatchEvent(keyEvent);
      }
    }
    
    
    setTimeout(() => {
      
      const sendButton = document.querySelector('#chat-submit, .cl-send-button, button[aria-label="Send message"]');
      
      if (sendButton && sendButton.disabled) {
        console.log("Botão ainda está desabilitado, tentando ativar...");
        
        
        sendButton.disabled = false;
        sendButton.removeAttribute('disabled');
        
        
        Object.defineProperty(sendButton, 'disabled', {
          value: false,
          writable: true,
          configurable: true
        });
        
        
        sendButton.classList.remove('disabled', 'pointer-events-none', 'opacity-50');
        
        
        const events = ['keydown', 'keyup', 'keypress', 'input', 'change', 'blur', 'focus'];
        events.forEach(eventType => {
          inputField.dispatchEvent(new Event(eventType, { bubbles: true }));
        });
      }
      
      
      setTimeout(() => {
        if (sendButton) {
          console.log("Clicando no botão de envio...");
          sendButton.click();
        } else {
          
          console.log("Botão não encontrado, tentando alternativas...");
          
          
          const enterEvent = new KeyboardEvent('keydown', {
            key: 'Enter',
            code: 'Enter',
            keyCode: 13,
            which: 13,
            bubbles: true,
            cancelable: true
          });
          inputField.dispatchEvent(enterEvent);
          
          
          const form = inputField.closest('form');
          if (form) {
            console.log("Enviando formulário...");
            form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
          }
        }
      }, 200);
    }, 300);
  }
  
  
  function debugChainlitInterface() {
    console.log("Depurando interface do Chainlit...");
    
    
    const inputs = document.querySelectorAll('input, textarea, [contenteditable="true"]');
    console.log("Campos de entrada encontrados:", inputs.length);
    inputs.forEach((input, i) => {
      console.log(`Input ${i}:`, input.tagName, input.className, input.id);
    });
    
    
    const buttons = document.querySelectorAll('button');
    console.log("Botões encontrados:", buttons.length);
    buttons.forEach((btn, i) => {
      console.log(`Botão ${i}:`, btn.textContent.trim(), btn.className, btn.getAttribute('aria-label'));
    });
    
    
    const forms = document.querySelectorAll('form');
    console.log("Formulários encontrados:", forms.length);
  }


document.addEventListener('DOMContentLoaded', () => {
  setTimeout(debugChainlitInterface, 2000); 
});

  
  function makeDraggableFast(element, handle) {
    let isDragging = false;
    let offsetX, offsetY;
    
    
    let lastX = 0, lastY = 0;
    let rafId = null;
    
    function updatePosition() {
      element.style.left = lastX + 'px';
      element.style.top = lastY + 'px';
      element.style.right = 'auto'; 
      element.style.bottom = 'auto';
      rafId = null;
    }
    
    handle.addEventListener('mousedown', function(e) {
      if (e.target === handle || e.target.parentNode === handle) {
        isDragging = true;
        
        
        const rect = element.getBoundingClientRect();
        offsetX = e.clientX - rect.left;
        offsetY = e.clientY - rect.top;
        
        
        element.style.transition = 'none';
        element.style.opacity = '0.9';
        
        e.preventDefault();
      }
    });

    document.addEventListener('mousemove', function(e) {
      if (!isDragging) return;
      
      
      lastX = e.clientX - offsetX;
      lastY = e.clientY - offsetY;
      
      
      const maxX = window.innerWidth - element.offsetWidth;
      const maxY = window.innerHeight - element.offsetHeight;
      
      lastX = Math.max(0, Math.min(lastX, maxX));
      lastY = Math.max(0, Math.min(lastY, maxY));
      
      
      if (!rafId) {
        rafId = requestAnimationFrame(updatePosition);
      }
    });
    
    document.addEventListener('mouseup', function() {
      if (isDragging) {
        isDragging = false;
        element.style.opacity = '1';
        
        
        if (rafId) {
          cancelAnimationFrame(rafId);
          updatePosition();
        }
      }
    });
    
    
    handle.addEventListener('touchstart', function(e) {
      if (e.target === handle || e.target.parentNode === handle) {
        isDragging = true;
        
        const rect = element.getBoundingClientRect();
        const touch = e.touches[0];
        
        offsetX = touch.clientX - rect.left;
        offsetY = touch.clientY - rect.top;
        
        element.style.transition = 'none';
        element.style.opacity = '0.9';
        
        e.preventDefault();
      }
    });
    
    document.addEventListener('touchmove', function(e) {
      if (!isDragging) return;
      
      const touch = e.touches[0];
      
      lastX = touch.clientX - offsetX;
      lastY = touch.clientY - offsetY;
      
      const maxX = window.innerWidth - element.offsetWidth;
      const maxY = window.innerHeight - element.offsetHeight;
      
      lastX = Math.max(0, Math.min(lastX, maxX));
      lastY = Math.max(0, Math.min(lastY, maxY));
      
      if (!rafId) {
        rafId = requestAnimationFrame(updatePosition);
      }
      
      e.preventDefault();
    });
    
    document.addEventListener('touchend', function() {
      if (isDragging) {
        isDragging = false;
        element.style.opacity = '1';
        
        if (rafId) {
          cancelAnimationFrame(rafId);
          updatePosition();
        }
      }
    });
  }
  function replaceMessageAvatars() {
    
    const avatars = document.querySelectorAll('.cl-avatar-container img, .cl-message-avatar img');
    
    avatars.forEach(avatar => {
      
      avatar.src = 'public/logo-pericial.png';
      avatar.srcset = ''; 
      avatar.style.width = '30px';
      avatar.style.height = '30px';
      
      
      avatar.setAttribute('data-original-replaced', 'true');
      
      
      if (!avatar.hasAttribute('protected')) {
        Object.defineProperty(avatar, 'src', {
          get: function() { return 'public/logo-pericial.png'; },
          set: function() { return 'public/logo-pericial.png'; },
          configurable: false
        });
        avatar.setAttribute('protected', 'true');
      }
    });
    
    
    const avatarContainers = document.querySelectorAll('.cl-avatar, .cl-message-avatar');
    avatarContainers.forEach(container => {
      container.style.backgroundImage = 'none';
    });
  }
  
  function initInterface() {

    replaceFavicon();
    
    applyThemeColors();
    
    
    createPromptPanel();
    
    
    document.title = "SISTEMA PERICIAL - Análise Forense";
    
    
    const observer = new MutationObserver(function(mutations) {
      if (!document.getElementById('prompt-panel')) {
        createPromptPanel();
      } 

      
      const themeChanged = mutations.some(mutation => {
        return mutation.target.classList && 
               (mutation.target.classList.contains('dark') || 
                mutation.target.classList.contains('light') ||
                mutation.target.hasAttribute('data-theme'));
      });
      
      if (themeChanged) {
        applyThemeColors();
      }

      removeChainlitReferences();
    });

    
  function removeChainlitReferences() {
    
    const footer = document.querySelector('.cl-footer');
    if (footer) {
      footer.style.display = 'none';
    }

  const chainlitAvatars = document.querySelectorAll('.cl-avatar-container img[src*="chainlit"]');
    chainlitAvatars.forEach(avatar => {
      avatar.src = '/logo.svg';
      avatar.style.width = '30px';
      avatar.style.height = '30px';
    });
  }

    observer.observe(document.documentElement, { 
      attributes: true,
      attributeFilter: ['class', 'data-theme']
    });
    
    observer.observe(document.body, { 
      attributes: true,
      attributeFilter: ['class', 'data-theme'],
      childList: true
    });
    
    
    if (window.matchMedia) {
      const colorSchemeQuery = window.matchMedia('(prefers-color-scheme: dark)');
      colorSchemeQuery.addEventListener('change', function() {
        applyThemeColors();
      });
    }
    
    
    addThemeToggleButton();
  }
  
  
  function addThemeToggleButton() {
    
    const existingButton = document.querySelector('[aria-label="Toggle theme"]');
    if (existingButton) {
      
      return;
    }
    
    
    const header = document.querySelector('.cl-header, header');
    if (!header) {
      return;
    }
    
    
    const themeButton = document.createElement('button');
    themeButton.setAttribute('aria-label', 'Toggle theme');
    themeButton.style.background = 'none';
    themeButton.style.border = 'none';
    themeButton.style.cursor = 'pointer';
    themeButton.style.padding = '8px';
    themeButton.style.borderRadius = '4px';
    themeButton.style.marginLeft = '8px';
    
    
    themeButton.innerHTML = detectCurrentTheme() === 'dark' ? '☀️' : '🌙';
    
    
    themeButton.addEventListener('click', function() {
      const currentTheme = detectCurrentTheme();
      const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
      
      
      this.innerHTML = newTheme === 'dark' ? '☀️' : '🌙';
      
      
      document.documentElement.setAttribute('data-theme', newTheme);
      document.documentElement.classList.remove(currentTheme);
      document.documentElement.classList.add(newTheme);
      
      
      applyThemeColors();
    });
    
    
    header.appendChild(themeButton);
  }
  
  
  setTimeout(initInterface, 500);
});