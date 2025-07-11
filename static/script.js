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
        
        const response = await fetch('/status'); // Changed from '/model-status' to '/status'
        if (!response.ok) {
            throw new Error('Failed to fetch model status');
        }
        
        const data = await response.json();
        
        // Update UI with model status
        modelStatus.textContent = data.status || 'Not Trained';
        
        // Set status color based on state
        if (data.status === 'Ready') {
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
    if (dropZone) {
        // Drag and drop events
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
        });
        
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, unhighlight, false);
        });
        
        dropZone.addEventListener('drop', handleDrop, false);
        
        // Click to select file
        dropZone.addEventListener('click', () => {
            fileInput.click();
        });
    }
    
    // File input change
    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
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
    
    // Reset any previous error states
    const errorInputs = document.querySelectorAll('.border-red-500, .border-green-500');
    errorInputs.forEach(input => {
        input.classList.remove('border-red-500', 'border-green-500');
    });
    
    // Check if model is ready
    try {
        const status = await checkModelStatus();
        if (status.status !== 'Ready') {
            showToast('Model is not ready. Please train the model first.', 'error');
            return;
        }
    } catch (error) {
        console.error('Error checking model status:', error);
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
    
    // Show loading state
    showLoading('Predicting salary...');
    
    try {
        // Prepare the request data with proper types
        const requestData = { ...data };
        
        // Ensure numeric fields are numbers
        const numericFields = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week'];
        numericFields.forEach(field => {
            if (field in requestData) {
                const numValue = parseFloat(requestData[field]);
                requestData[field] = isNaN(numValue) ? 0 : numValue;
            }
        });
        
        // Make the prediction request
        const response = await fetch('/predict', {
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
    const resultDiv = document.getElementById('result');
    const confidence = result.confidence ? Math.round(result.confidence * 100) : 0;
    const isHighIncome = (result.prediction && (result.prediction.toString().includes('>50K') || result.prediction.toString() === '1')) || 
                        (result.class !== undefined && (result.class === 1 || result.class === '>50K'));
    
    // Ensure we have valid confidence value
    const validConfidence = Math.min(Math.max(confidence, 0), 100);
    
    // Create a more informative result display
    resultDiv.innerHTML = `
        <div class="prediction-result p-6 rounded-lg ${isHighIncome ? 'bg-green-50' : 'bg-blue-50'} border ${isHighIncome ? 'border-green-200' : 'border-blue-200'} shadow-sm">
            <div class="text-center">
                <div class="inline-flex items-center justify-center w-20 h-20 rounded-full ${isHighIncome ? 'bg-green-100 text-green-600' : 'bg-blue-100 text-blue-600'} mb-4 shadow-sm">
                    <i class="fas ${isHighIncome ? 'fa-dollar-sign' : 'fa-euro-sign'} text-3xl"></i>
                </div>
                <h3 class="text-2xl font-bold ${isHighIncome ? 'text-green-800' : 'text-blue-800'} mb-2">
                    ${isHighIncome ? 'High Income (>$50K)' : 'Low Income (≤$50K)'}
                </h3>
                
                <div class="mb-4">
                    <div class="flex justify-between text-sm text-gray-600 mb-1">
                        <span>Confidence Level:</span>
                        <span class="font-medium">${validConfidence}%</span>
                    </div>
                    <div class="w-full bg-gray-200 rounded-full h-2.5">
                        <div class="h-2.5 rounded-full ${isHighIncome ? 'bg-green-500' : 'bg-blue-500'}" 
                             style="width: ${validConfidence}%"></div>
                    </div>
                </div>
                
                ${validConfidence < 60 ? `
                <div class="bg-yellow-50 border-l-4 border-yellow-400 p-3 mb-4 rounded">
                    <div class="flex">
                        <div class="flex-shrink-0">
                            <i class="fas fa-exclamation-triangle text-yellow-500"></i>
                        </div>
                        <div class="ml-3">
                            <p class="text-sm text-yellow-700">
                                Note: This is a low confidence prediction. The model is less certain about this result.
                            </p>
                        </div>
                    </div>
                </div>` : ''}
                
                <div class="mt-4 pt-4 border-t border-gray-200">
                    <p class="text-sm text-gray-600">
                        <i class="fas fa-info-circle text-gray-400 mr-1"></i>
                        Prediction based on the provided employee details
                    </p>
                </div>
            </div>
        </div>
    `;
    
    // Add animation class
    resultDiv.classList.add('animate-fade-in');
    
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

    // Show required columns to user
    const requiredColumns = [
        'age', 'workclass', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race',
        'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
        'native-country', 'income'
    ];

    const formData = new FormData();
    formData.append('file', file);

    showLoading('Training model...', 'Please wait while we process your data');
    
    try {
        const response = await fetch('/train', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (!response.ok) {
            let errorMsg = result.error || 'Failed to train model';
            if (result.details) {
                errorMsg += `: ${result.details}`;
            }
            throw new Error(errorMsg);
        }

        showToast('Model trained successfully!', 'success');
        showModal(false);
        checkModelStatus();
    } catch (error) {
        console.error('Upload error:', error);
        let errorMsg = error.message;
        
        // Provide more user-friendly error messages
        if (error.message.includes('Missing required columns')) {
            errorMsg = 'Invalid CSV format. Please ensure your file includes all required columns.';
            // Show the required columns to the user
            showToast(errorMsg, 'error');
            const columnsList = requiredColumns.map(col => `<li>${col}</li>`).join('');
            alert(`Required columns are:\n${requiredColumns.join('\n')}`);
        } else if (error.message.includes('Failed to fetch')) {
            errorMsg = 'Failed to connect to the server. Please check your internet connection.';
            showToast(errorMsg, 'error');
        } else {
            showToast(errorMsg, 'error');
        }
    } finally {
        hideLoading();
    }
}

// ... rest of the code remains the same ...
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
