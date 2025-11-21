// API configuration - will be set from environment variable or default to relative path
const API_BASE_URL = (window as any).API_BASE_URL || '';

// Type definitions
interface Source {
    title: string;
    link: string;
    score: number;
}

interface QueryResponse {
    answer: string;
    sources: Source[];
    question: string;
}

interface ErrorResponse {
    detail: string;
}

// DOM elements
const queryForm = document.getElementById('queryForm') as HTMLFormElement;
const questionInput = document.getElementById('questionInput') as HTMLInputElement;
const submitBtn = document.getElementById('submitBtn') as HTMLButtonElement;
const kSelect = document.getElementById('kSelect') as HTMLSelectElement;
const tempSlider = document.getElementById('tempSlider') as HTMLInputElement;
const tempValue = document.getElementById('tempValue') as HTMLSpanElement;
const resultsSection = document.getElementById('resultsSection') as HTMLDivElement;
const errorSection = document.getElementById('errorSection') as HTMLDivElement;
const displayedQuestion = document.getElementById('displayedQuestion') as HTMLParagraphElement;
const answerContent = document.getElementById('answerContent') as HTMLDivElement;
const sourcesList = document.getElementById('sourcesList') as HTMLDivElement;
const errorText = document.getElementById('errorText') as HTMLParagraphElement;

// Temperature slider handler
if (tempSlider && tempValue) {
    tempSlider.addEventListener('input', (e: Event) => {
        const target = e.target as HTMLInputElement;
        const value = parseFloat(target.value) / 10;
        tempValue.textContent = value.toFixed(1);
    });
}

// Form submission handler
if (queryForm) {
    queryForm.addEventListener('submit', async (e: Event) => {
        e.preventDefault();
        
        const question = questionInput.value.trim();
        if (!question) return;

        // Hide previous results/errors
        if (resultsSection) resultsSection.style.display = 'none';
        if (errorSection) errorSection.style.display = 'none';

        // Show loading state
        setLoadingState(true);

        try {
            const k = parseInt(kSelect.value);
            const temperature = parseFloat(tempSlider.value) / 10;

            // Normalize API URL - remove trailing slash to avoid double slashes
            const apiUrl = (API_BASE_URL || window.location.origin).replace(/\/+$/, '');
            const response = await fetch(`${apiUrl}/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question: question,
                    k: k,
                    temperature: temperature,
                }),
            });

            if (!response.ok) {
                const errorData: ErrorResponse = await response.json().catch(() => ({ 
                    detail: 'Unknown error occurred' 
                }));
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }

            const data: QueryResponse = await response.json();
            displayResults(data);

        } catch (error) {
            console.error('Error:', error);
            const errorMessage = error instanceof Error 
                ? error.message 
                : 'Failed to fetch results. Please try again.';
            displayError(errorMessage);
        } finally {
            setLoadingState(false);
        }
    });
}

function setLoadingState(loading: boolean): void {
    if (submitBtn) submitBtn.disabled = loading;
    if (questionInput) questionInput.disabled = loading;
    if (kSelect) kSelect.disabled = loading;
    if (tempSlider) tempSlider.disabled = loading;
    
    const btnText = submitBtn?.querySelector('.btn-text') as HTMLElement;
    const btnLoader = submitBtn?.querySelector('.btn-loader') as HTMLElement;
    
    if (loading) {
        if (btnText) btnText.style.display = 'none';
        if (btnLoader) btnLoader.style.display = 'inline';
    } else {
        if (btnText) btnText.style.display = 'inline';
        if (btnLoader) btnLoader.style.display = 'none';
    }
}

function displayResults(data: QueryResponse): void {
    if (!displayedQuestion || !answerContent || !sourcesList || !resultsSection) return;

    // Display question
    displayedQuestion.textContent = data.question;

    // Display answer
    answerContent.textContent = data.answer;

    // Display sources
    sourcesList.innerHTML = '';
    
    if (data.sources && data.sources.length > 0) {
        data.sources.forEach((source: Source, index: number) => {
            const sourceItem = document.createElement('div');
            sourceItem.className = 'source-item';
            
            const sourceInfo = document.createElement('div');
            sourceInfo.className = 'source-info';
            
            const sourceTitle = document.createElement('div');
            sourceTitle.className = 'source-title';
            sourceTitle.textContent = `${index + 1}. ${source.title}`;
            
            const sourceLink = document.createElement('a');
            sourceLink.className = 'source-link';
            sourceLink.href = source.link;
            sourceLink.target = '_blank';
            sourceLink.rel = 'noopener noreferrer';
            sourceLink.textContent = 'View Post →';
            
            const sourceScore = document.createElement('div');
            sourceScore.className = 'source-score';
            sourceScore.textContent = `Score: ${source.score.toFixed(4)}`;
            
            sourceInfo.appendChild(sourceTitle);
            sourceInfo.appendChild(sourceLink);
            sourceItem.appendChild(sourceInfo);
            sourceItem.appendChild(sourceScore);
            
            sourcesList.appendChild(sourceItem);
        });
    } else {
        const noSources = document.createElement('div');
        noSources.className = 'source-item';
        noSources.textContent = 'No sources found.';
        sourcesList.appendChild(noSources);
    }

    // Show results section
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function displayError(message: string): void {
    if (!errorText || !errorSection) return;
    errorText.textContent = message;
    errorSection.style.display = 'block';
    errorSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Allow Enter key to submit (form already handles this, but ensure it works)
if (questionInput && queryForm) {
    questionInput.addEventListener('keydown', (e: KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            queryForm.dispatchEvent(new Event('submit'));
        }
    });
}

// Accordion functionality for About section
const aboutHeader = document.getElementById('aboutHeader') as HTMLElement;
const aboutContent = document.getElementById('aboutContent') as HTMLElement;

if (aboutHeader && aboutContent) {
    aboutHeader.addEventListener('click', () => {
        const isExpanded = aboutHeader.classList.contains('expanded');
        
        if (isExpanded) {
            aboutHeader.classList.remove('expanded');
            aboutContent.classList.remove('expanded');
        } else {
            aboutHeader.classList.add('expanded');
            aboutContent.classList.add('expanded');
        }
    });
}

