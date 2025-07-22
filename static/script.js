// DOM Elements
const predictionForm = document.getElementById('predictionForm');
const uploadBtn = document.getElementById('uploadBtn');
const closeModal = document.getElementById('closeModal');
const uploadModal = document.getElementById('uploadModal');
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const fileName = document.getElementById('fileName');
const uploadSubmit = document.getElementById('uploadSubmit');
const loadingOverlay = document.getElementById('loadingOverlay');
const loadingText = document.getElementById('loadingText');
const loadingSubtext = document.getElementById('loadingSubtext');
const progressBar = document.getElementById('progressBar');
const progressBarFill = document.getElementById('progressBarFill');
const toast = document.getElementById('toast');
const toastMessage = document.getElementById('toastMessage');
const toastIcon = document.getElementById('toastIcon');
const modelStatus = document.getElementById('modelStatus');
const trainedOn = document.getElementById('trainedOn');
const modelAccuracy = document.getElementById('modelAccuracy');
const retrainCheckbox = document.getElementById('retrainCheckbox');
const trainingOptions = document.getElementById('trainingOptions');

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    checkModelStatus();
    setupEventListeners();
});

// Check model status on page load and periodically
let modelCheckInterval;

async function checkModelStatus() {
    try {
        clearInterval(modelCheckInterval);
        
        const response = await fetch('/api/model-status');
        if (!response.ok) {
            throw new Error('Failed to fetch model status');
        }
        
        const data = await response.json();
        
        // Update UI with model status
        modelStatus.textContent = data.status || 'Not Trained';
        
        // Set status color based on state
        if (data.status && data.status.startsWith('Ready')) {
            modelStatus.className = 'text-green-600 font-medium';
            // Only start checking periodically if model is ready
            modelCheckInterval = setInterval(checkModelStatus, 30000); // Check every 30 seconds
        } else if (data.status === 'Not Trained') {
            modelStatus.className = 'text-yellow-600 font-medium';
            // Check more frequently if not trained
            modelCheckInterval = setInterval(checkModelStatus, 10000); // Check every 10 seconds
        } else {
            modelStatus.className = 'text-red-600 font-medium';
            // Check more frequently if error
            modelCheckInterval = setInterval(checkModelStatus, 5000); // Check every 5 seconds
        }
        
        // Update training date if available
        if (data.trained_on) {
            trainedOn.textContent = new Date(data.trained_on).toLocaleString();
        } else {
            trainedOn.textContent = 'Never';
        }
        
        // Update accuracy if available
        if (data.accuracy !== undefined && data.accuracy !== null) {
            modelAccuracy.textContent = `${(data.accuracy * 100).toFixed(2)}%`;
        } else {
            modelAccuracy.textContent = 'N/A';
        }
        
        return data;
    } catch (error) {
        console.error('Error checking model status:', error);
        modelStatus.textContent = 'Error';
        modelStatus.className = 'text-red-600 font-medium';
        trainedOn.textContent = 'N/A';
        modelAccuracy.textContent = 'N/A';
        
        // Retry after delay
        setTimeout(checkModelStatus, 5000);
        return { status: 'Error' };
    }
}

// Set up event listeners
function setupEventListeners() {
    // Form submission
    if (predictionForm) {
        predictionForm.addEventListener('submit', handlePrediction);
    }
    
    // Modal controls
    if (uploadBtn) {
        uploadBtn.addEventListener('click', () => showModal(true));
    }
    
    if (closeModal) {
        closeModal.addEventListener('click', () => showModal(false));
    }
    
    // File upload handling
    if (dropZone && fileInput) {
        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
            document.body.addEventListener(eventName, preventDefaults, false);
        });
        
        // Highlight drop zone when item is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, highlight, false);
        });
        
        // Remove highlight when item leaves drop zone
        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, unhighlight, false);
        });
        
        // Handle dropped files
        dropZone.addEventListener('drop', handleDrop, false);
        
        // Handle file selection via the file input
        fileInput.addEventListener('change', handleFileSelect, false);
    }
    
    // Upload submit button
    if (uploadSubmit) {
        uploadSubmit.addEventListener('click', handleUpload);
    }
    
    // Retrain checkbox
    if (retrainCheckbox) {
        retrainCheckbox.addEventListener('change', (e) => {
            trainingOptions.classList.toggle('hidden', !e.target.checked);
        });
    }
}

// Handle form submission for prediction
async function handlePrediction(e) {
    e.preventDefault();
    
    // Show loading state immediately for better UX
    showLoading('Predicting salary...');
    
    // Reset any previous error states
    const errorInputs = document.querySelectorAll('.border-red-500, .border-green-500');
    errorInputs.forEach(input => {
        input.classList.remove('border-red-500', 'border-green-500');
    });
    
    // Check if model is ready (after showing loader)
    try {
        const status = await checkModelStatus();
        if (!status.status || !status.status.startsWith('Ready')) {
            hideLoading(); // Hide loader if model is not ready
            showToast('Model is not ready. Please train the model first.', 'error');
            return;
        }
    } catch (error) {
        console.error('Error checking model status:', error);
        hideLoading(); // Hide loader on error
        showToast('Failed to check model status. Please try again.', 'error');
        return;
    }
    
    // Get all form inputs and prepare data
    const formInputs = predictionForm.querySelectorAll('input:not([type="button"]), select, textarea');
    const data = {};
    const validationErrors = [];
    let hasValidationErrors = false;
    
    // Field validation rules
    const validationRules = {
        'age': { min: 15, max: 100, required: true },
        'fnlwgt': { min: 10000, max: 2000000, required: true },
        'educational-num': { min: 1, max: 16, required: true },
        'capital-gain': { min: 0, required: true },
        'capital-loss': { min: 0, required: true },
        'hours-per-week': { min: 1, max: 99, required: true },
        'workclass': { required: true },
        'education': { required: true },
        'marital-status': { required: true },
        'occupation': { required: true },
        'relationship': { required: true },
        'race': { required: true },
        'gender': { required: true },
        'native-country': { required: true }
    };
    
    // Process and validate form data
    formInputs.forEach(input => {
        const name = input.name;
        if (!name) return; // Skip unnamed inputs
        
        let value = input.value.trim();
        const rules = validationRules[name] || {};
        const inputWrapper = input.closest('.form-group') || input.closest('div') || input;
        
        // Reset visual feedback
        inputWrapper.classList.remove('border-red-500', 'border-green-500');
        
        // Handle different input types
        if (input.type === 'number') {
            value = value === '' ? NaN : parseFloat(value);
            if (isNaN(value) && rules.required) {
                validationErrors.push(`Please enter a valid number for ${input.placeholder || name}`);
                hasValidationErrors = true;
                inputWrapper.classList.add('border-red-500');
                return;
            }
        } else if (input.type === 'checkbox') {
            value = input.checked;
        } else if (input.type === 'select-one') {
            if (rules.required && (!value || value === 'Select an option')) {
                validationErrors.push(`Please select a value for ${input.placeholder || name}`);
                hasValidationErrors = true;
                inputWrapper.classList.add('border-red-500');
                return;
            }
        }
        
        // Check required fields
        if (rules.required && (value === '' || value === null || value === undefined)) {
            validationErrors.push(`${input.placeholder || name} is required`);
            hasValidationErrors = true;
            inputWrapper.classList.add('border-red-500');
            return;
        }
        
        // Numeric validation
        if (typeof value === 'number' && rules) {
            if (rules.min !== undefined && value < rules.min) {
                validationErrors.push(`${input.placeholder || name} must be at least ${rules.min}`);
                hasValidationErrors = true;
                inputWrapper.classList.add('border-red-500');
                return;
            }
            if (rules.max !== undefined && value > rules.max) {
                validationErrors.push(`${input.placeholder || name} cannot exceed ${rules.max}`);
                hasValidationErrors = true;
                inputWrapper.classList.add('border-red-500');
                return;
            }
        }
        
        // If we made it here, validation passed for this field
        inputWrapper.classList.add('border-green-500');
        data[name] = value;
    });
    
    // Show all validation errors if any
    if (hasValidationErrors) {
        showToast(validationErrors[0] || 'Please check the form for errors', 'error');
        
        // If there's only one error, scroll to it
        if (validationErrors.length === 1) {
            const firstError = document.querySelector('.border-red-500');
            if (firstError) {
                firstError.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        }
        return;
    }
    
    try {
        // Map form field names to model feature names
        const fieldMappings = {
            'age': 'age',
            'workclass': 'workclass',
            'fnlwgt': 'fnlwgt',
            'education': 'education',
            'educational-num': 'education_num',
            'marital-status': 'marital_status',
            'occupation': 'occupation',
            'relationship': 'relationship',
            'race': 'race',
            'gender': 'sex',
            'capital-gain': 'capital_gain',
            'capital-loss': 'capital_loss',
            'hours-per-week': 'hours_per_week',
            'native-country': 'native_country'
        };
        
        // Prepare the request data with proper types and mapped field names
        const requestData = {};
        
        Object.keys(data).forEach(field => {
            const modelField = fieldMappings[field] || field;
            let value = data[field];
            
            // Convert numeric fields to numbers
            const numericFields = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week'];
            if (numericFields.includes(modelField)) {
                const numValue = parseFloat(value);
                requestData[modelField] = isNaN(numValue) ? 0 : numValue;
            } else {
                requestData[modelField] = value;
            }
        });
        
        // Make the prediction request
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || 'Failed to get prediction. Please try again.');
        }
        
        const result = await response.json();
        
        // Validate the response
        if (!result || (result.prediction === undefined && result.class === undefined)) {
            throw new Error('Invalid response from server');
        }
        
        // Display the prediction result
        displayPredictionResult(result);
        
        // Log the successful prediction
        console.log('Prediction successful:', result);
        
    } catch (error) {
        console.error('Prediction error:', error);
        
        // More specific error messages based on error type
        let errorMessage = 'Failed to get prediction. ';
        if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
            errorMessage += 'Please check your internet connection.';
        } else {
            errorMessage += error.message || 'Please try again later.';
        }
        
        showToast(errorMessage, 'error');
        
        // Show error in the UI
        const resultDiv = document.getElementById('result');
        resultDiv.innerHTML = `
            <div class="bg-red-50 border-l-4 border-red-500 p-4 mb-4 rounded">
                <div class="flex">
                    <div class="flex-shrink-0">
                        <i class="fas fa-exclamation-circle text-red-500"></i>
                    </div>
                    <div class="ml-3">
                        <h3 class="text-sm font-medium text-red-800">Prediction Failed</h3>
                        <div class="mt-2 text-sm text-red-700">
                            <p>${errorMessage}</p>
                            <p class="mt-2">Please check your input and try again.</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
    } finally {
        hideLoading();
    }
}

// Display prediction result
function displayPredictionResult(result) {
    const initialPlaceholder = document.getElementById('initialResultPlaceholder');
    const predictionResultArea = document.getElementById('predictionResultArea');
    const predictionTextDiv = document.getElementById('predictionText');
    const predictionDetailsDiv = document.getElementById('predictionDetails');
    const chartCanvas = document.getElementById('confidenceChart');
    const resultDiv = document.getElementById('result'); // Add missing resultDiv reference
    
    const confidence = result.confidence ? Math.round(result.confidence * 100) : 0;
    const isHighIncome = (result.prediction && result.prediction.includes('>50K')) || result.class === 1;

    // Hide initial placeholder and show the result area
    initialPlaceholder.classList.add('hidden');
    predictionResultArea.classList.remove('hidden');

    // Update the text inside the donut chart
    const incomeLabel = isHighIncome ? '>$50K' : '<=$50K';
    const incomeTextColor = isHighIncome ? 'text-green-700' : 'text-blue-700';
    predictionTextDiv.innerHTML = `
        <span class="text-3xl font-bold ${incomeTextColor}">${confidence}%</span>
        <span class="text-sm ${incomeTextColor}">Confidence</span>
    `;

    // Update the details below the chart
    const detailsTextColor = isHighIncome ? 'text-green-800' : 'text-blue-800';
    const detailsBgColor = isHighIncome ? 'bg-green-50' : 'bg-blue-50';
    predictionDetailsDiv.innerHTML = `
        <div class="p-3 rounded-lg ${detailsBgColor} text-center">
            <p class="font-semibold ${detailsTextColor}">Predicted Salary Range</p>
            <p class="text-xl font-bold ${detailsTextColor}">${incomeLabel}</p>
        </div>
        ${confidence < 60 ? `
        <div class="mt-3 bg-yellow-50 border-l-4 border-yellow-400 p-3 rounded">
            <div class="flex">
                <div class="flex-shrink-0">
                    <i class="fas fa-exclamation-triangle text-yellow-500"></i>
                </div>
                <div class="ml-3">
                    <p class="text-sm text-yellow-700">
                        This is a low confidence prediction. The model is less certain about this result.
                    </p>
                </div>
            </div>
        </div>` : ''}
    `;

    // Create or update the donut chart
    const chartData = {
        datasets: [{
            data: [confidence, 100 - confidence],
            backgroundColor: [
                isHighIncome ? 'rgba(5, 150, 105, 1)' : 'rgba(59, 130, 246, 1)',
                'rgba(229, 231, 235, 1)'
            ],
            borderColor: '#fff',
            borderWidth: 2,
            hoverBorderColor: '#fff'
        }]
    };

    // Check if a chart instance already exists and destroy it
    if (window.myConfidenceChart) {
        window.myConfidenceChart.destroy();
    }

    // Create the new chart
    window.myConfidenceChart = new Chart(chartCanvas, {
        type: 'doughnut',
        data: chartData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '80%',
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    enabled: false
                }
            },
            animation: {
                animateScale: true,
                animateRotate: true
            }
        }
    });

    // Add animation class to the result area for a smooth fade-in
    predictionResultArea.classList.add('animate-fade-in');
    setTimeout(() => {
        predictionResultArea.classList.remove('animate-fade-in');
    }, 500);

    
    // Scroll to result with smooth animation
    setTimeout(() => {
        resultDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 100);
    
    // Remove animation class after it completes
    setTimeout(() => {
        resultDiv.classList.remove('animate-fade-in');
    }, 1000);
}

// Handle file upload
async function handleUpload() {
    const file = fileInput.files[0];
    if (!file) {
        showToast('Please select a file first', 'error');
        return;
    }
    
    // Check file type
    if (!file.name.endsWith('.csv')) {
        showToast('Please upload a CSV file with .csv extension', 'error');
        return;
    }
    
    // Show loading state with progress
    showLoading('Uploading and processing file...', 'This may take a few moments');
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/api/train', {
            method: 'POST',
            body: formData,
            headers: {
                'Accept': 'application/json'
            }
        });
        
        let result;
        try {
            // Try to parse as JSON
            result = await response.json();
        } catch (jsonError) {
            console.error('Failed to parse JSON response:', jsonError);
            throw new Error('Server returned an invalid response. Please try again.');
        }
        
        // Check if the response indicates an error
        if (!response.ok || result.status === 'error') {
            const errorMessage = result.message || `Server error: ${response.status} ${response.statusText}`;
            throw new Error(errorMessage);
        }
        
        // If we get here, the request was successful
        showToast(result.message || 'Model trained successfully!', 'success');
        
        // Reset file input
        fileInput.value = '';
        if (fileName) {
            fileName.textContent = 'No file selected';
        }
        
        // Update model status immediately with the new data
        if (result.trained_on) {
            modelStatus.textContent = 'Ready';
            modelStatus.className = 'text-green-600 font-medium';
            trainedOn.textContent = new Date(result.trained_on).toLocaleString();
            if (result.accuracy !== undefined) {
                modelAccuracy.textContent = `${(result.accuracy * 100).toFixed(2)}%`;
            }
        }
        
        // Close modal after a short delay
        setTimeout(() => {
            showModal(false);
            // Force a full status check to ensure everything is in sync
            checkModelStatus();
        }, 1500);
        
    } catch (error) {
        console.error('Upload error:', error);
        
        // More specific error messages
        let errorMessage = error.message || 'Failed to process file. Please try again.';
        
        if (error.message.includes('Failed to fetch')) {
            errorMessage = 'Unable to connect to the server. Please check your connection.';
        } else if (error.message.includes('Unexpected token')) {
            errorMessage = 'Server returned an invalid response. Please try again.';
        } else if (error.message.includes('NetworkError')) {
            errorMessage = 'Network error. Please check your internet connection.';
        }
        
        showToast(errorMessage, 'error');
    } finally {
        hideLoading();
    }
}

// Handle drag and drop
function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlight() {
    dropZone.classList.add('border-indigo-500', 'bg-indigo-50');
}

function unhighlight() {
    dropZone.classList.remove('border-indigo-500', 'bg-indigo-50');
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    
    if (files.length) {
        fileInput.files = files;
        handleFileSelect({ target: fileInput });
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    
    if (files.length) {
        const file = files[0];
        fileName.textContent = file.name;
        uploadSubmit.disabled = false;
    }
}

// Modal controls
function showModal(show) {
    if (show) {
        uploadModal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
    } else {
        uploadModal.classList.add('hidden');
        document.body.style.overflow = '';
        // Reset form
        if (fileInput) fileInput.value = '';
        if (fileName) fileName.textContent = 'No file selected';
        if (uploadSubmit) uploadSubmit.disabled = true;
    }
}

// Loading overlay
function showLoading(message = 'Processing...', submessage = 'This may take a few moments') {
    loadingText.textContent = message;
    loadingSubtext.textContent = submessage;
    loadingOverlay.classList.remove('hidden');
    progressBar.classList.add('hidden');
    progressBarFill.style.width = '0%';
}

function updateProgress(percent) {
    progressBar.classList.remove('hidden');
    progressBarFill.style.width = `${percent}%`;
}

function hideLoading() {
    loadingOverlay.classList.add('hidden');
}

// Toast notification
function showToast(message, type = 'success') {
    // Set icon and styles based on type
    let iconClass = 'fa-check-circle';
    let bgColor = 'bg-green-500';
    
    if (type === 'error') {
        iconClass = 'fa-exclamation-circle';
        bgColor = 'bg-red-500';
    } else if (type === 'warning') {
        iconClass = 'fa-exclamation-triangle';
        bgColor = 'bg-yellow-500';
    } else if (type === 'info') {
        iconClass = 'fa-info-circle';
        bgColor = 'bg-blue-500';
    }
    
    // Update toast content
    toastMessage.textContent = message;
    toastIcon.className = `fas ${iconClass} mr-2`;
    
    // Show toast
    toast.classList.remove('translate-y-16', 'opacity-0');
    toast.classList.add('-translate-y-0', 'opacity-100');
    
    // Hide after delay
    setTimeout(() => {
        toast.classList.add('translate-y-16', 'opacity-0');
        toast.classList.remove('-translate-y-0', 'opacity-100');
    }, 5000);
}

// Add model status endpoint
if (window.location.pathname === '/model-status') {
    fetch('/model-status')
        .then(response => response.json())
        .then(data => {
            console.log('Model status:', data);
        });
}

// Add a simple animation to form inputs
const inputs = document.querySelectorAll('input, select, textarea');
inputs.forEach(input => {
    input.addEventListener('focus', () => {
        input.parentElement.classList.add('ring-2', 'ring-indigo-200', 'rounded-lg');
    });
    
    input.addEventListener('blur', () => {
        input.parentElement.classList.remove('ring-2', 'ring-indigo-200', 'rounded-lg');
    });
});
