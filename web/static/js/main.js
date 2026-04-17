/**
 * Drug Advisor - Main JavaScript
 */

document.addEventListener('DOMContentLoaded', function() {
    
    // Initialize all components
    initCharacterCounter();
    initFormHandling();
    initTooltips();
    initSmoothScroll();
    initExampleButtons();
});

/**
 * Character counter for textarea
 */
function initCharacterCounter() {
    const textarea = document.getElementById('symptoms');
    if (!textarea) return;
    
    const counter = createCounter();
    textarea.parentNode.insertBefore(counter, textarea.nextSibling);
    
    function createCounter() {
        const div = document.createElement('div');
        div.className = 'char-counter';
        div.id = 'char-counter';
        return div;
    }
    
    function updateCounter() {
        const length = textarea.value.length;
        const counter = document.getElementById('char-counter');
        counter.textContent = `${length} characters (minimum 10)`;
        
        if (length < 10) {
            counter.classList.add('warning');
        } else {
            counter.classList.remove('warning');
        }
    }
    
    textarea.addEventListener('input', updateCounter);
    updateCounter();
}

/**
 * Form submission handling with loading state
 */
function initFormHandling() {
    const forms = document.querySelectorAll('form');
    
    forms.forEach(form => {
        form.addEventListener('submit', function() {
            const submitBtn = this.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.disabled = true;
                const originalText = submitBtn.innerHTML;
                submitBtn.setAttribute('data-original-text', originalText);
                submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Analyzing...';
            }
        });
    });
}

/**
 * Initialize Bootstrap tooltips
 */
function initTooltips() {
    if (typeof bootstrap === 'undefined') return;
    
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Smooth scroll for anchor links
 */
function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            if (href === '#') return;
            
            e.preventDefault();
            const target = document.querySelector(href);
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

/**
 * Example buttons for quick symptom entry
 */
function initExampleButtons() {
    const buttons = document.querySelectorAll('[data-example]');
    
    buttons.forEach(btn => {
        btn.addEventListener('click', function() {
            const textarea = document.getElementById('symptoms');
            if (textarea) {
                textarea.value = this.getAttribute('data-example');
                textarea.focus();
                
                // Trigger input event to update counter
                textarea.dispatchEvent(new Event('input'));
            }
        });
    });
}

/**
 * Copy example text to textarea
 * @param {string} text - The example text to copy
 */
function useExample(text) {
    const textarea = document.getElementById('symptoms');
    if (textarea) {
        textarea.value = text;
        textarea.focus();
        textarea.dispatchEvent(new Event('input'));
    }
}

/**
 * Show loading overlay
 */
function showLoading() {
    const overlay = document.createElement('div');
    overlay.className = 'spinner-overlay';
    overlay.innerHTML = `
        <div class="spinner-border text-primary" style="width: 3rem; height: 3rem;" role="status">
            <span class="visually-hidden">Loading...</span>
        </div>
    `;
    document.body.appendChild(overlay);
}

/**
 * Hide loading overlay
 */
function hideLoading() {
    const overlay = document.querySelector('.spinner-overlay');
    if (overlay) {
        overlay.remove();
    }
}

/**
 * Format number with commas
 * @param {number} num - Number to format
 * @returns {string} Formatted number
 */
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

/**
 * Calculate percentage
 * @param {number} value - Current value
 * @param {number} total - Total value
 * @returns {number} Percentage
 */
function calculatePercentage(value, total) {
    if (total === 0) return 0;
    return (value / total) * 100;
}

/**
 * Debounce function for performance
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time in milliseconds
 * @returns {Function} Debounced function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}