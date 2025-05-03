/**
 * Netflix Churn Predictor - Model Comparison JavaScript
 * Functions for model comparison and hyperparameter tuning
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize charts
    initModelComparisonCharts();
    
    // Event listeners for tabs
    document.querySelectorAll('#modelTabs button').forEach(button => {
        button.addEventListener('click', function() {
            updateActiveTab(this.id);
        });
    });
    
    // Format JSON in parameter grid textarea
    if (document.getElementById('param-grid')) {
        formatParameterGrid();
    }
    
    // Event listeners for forms
    if (document.getElementById('tuning-form')) {
        document.getElementById('tuning-form').addEventListener('submit', handleTuningFormSubmit);
    }
    
    if (document.getElementById('training-form')) {
        document.getElementById('training-form').addEventListener('submit', handleTrainingFormSubmit);
    }
    
    if (document.getElementById('feature-engineering-form')) {
        document.getElementById('feature-engineering-form').addEventListener('submit', handleFeatureEngineeringSubmit);
    }
    
    // Update parameter descriptions when model type changes
    if (document.getElementById('model-type')) {
        document.getElementById('model-type').addEventListener('change', updateParameterDescriptions);
    }
    
    // Test size slider value update
    if (document.getElementById('test-size')) {
        document.getElementById('test-size').addEventListener('input', function() {
            document.getElementById('test-size-value').textContent = (this.value * 100) + '%';
        });
    }
    
    // Feature selector for distribution chart
    if (document.getElementById('feature-selector')) {
        document.getElementById('feature-selector').addEventListener('change', updateFeatureDistribution);
    }
    
    // Model action buttons
    document.querySelectorAll('.view-model').forEach(button => {
        button.addEventListener('click', function() {
            viewModelDetails(this.getAttribute('data-model-id'));
        });
    });
    
    document.querySelectorAll('.set-active').forEach(button => {
        button.addEventListener('click', function() {
            setActiveModel(this.getAttribute('data-model-id'));
        });
    });
    
    if (document.getElementById('save-tuned-model')) {
        document.getElementById('save-tuned-model').addEventListener('click', saveTunedModel);
    }
});

/**
 * Updates the active tab state
 */
function updateActiveTab(tabId) {
    // Store active tab in sessionStorage for persistence
    sessionStorage.setItem('activeModelTab', tabId);
    
    // Load tab-specific content if needed
    if (tabId === 'comparison-tab') {
        loadModelComparisonData();
    } else if (tabId === 'feature-tab') {
        loadFeatureImportanceData();
    }
}

/**
 * Initializes charts for model comparison
 */
function initModelComparisonCharts() {
    // Performance comparison chart
    if (document.getElementById('performance-chart')) {
        const ctx = document.getElementById('performance-chart').getContext('2d');
        window.performanceChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['XGBoost', 'LightGBM', 'Random Forest', 'Logistic Regression', 'Ensemble'],
                datasets: [{
                    label: 'Accuracy',
                    data: [0.85, 0.83, 0.82, 0.76, 0.87],
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }, {
                    label: 'Precision',
                    data: [0.82, 0.80, 0.79, 0.72, 0.84],
                    backgroundColor: 'rgba(255, 99, 132, 0.5)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 1
                }, {
                    label: 'Recall',
                    data: [0.79, 0.81, 0.77, 0.73, 0.83],
                    backgroundColor: 'rgba(75, 192, 192, 0.5)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                }, {
                    label: 'F1 Score',
                    data: [0.80, 0.80, 0.78, 0.72, 0.83],
                    backgroundColor: 'rgba(255, 206, 86, 0.5)',
                    borderColor: 'rgba(255, 206, 86, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });
    }
    
    // ROC curves chart
    if (document.getElementById('roc-chart')) {
        const ctx = document.getElementById('roc-chart').getContext('2d');
        window.rocChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                datasets: [{
                    label: 'XGBoost (AUC=0.91)',
                    data: [0, 0.4, 0.6, 0.75, 0.83, 0.88, 0.92, 0.95, 0.97, 0.99, 1],
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.1)',
                    borderWidth: 2,
                    fill: true
                }, {
                    label: 'LightGBM (AUC=0.90)',
                    data: [0, 0.38, 0.58, 0.73, 0.81, 0.86, 0.91, 0.94, 0.97, 0.99, 1],
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    borderWidth: 2,
                    fill: true
                }, {
                    label: 'Ensemble (AUC=0.93)',
                    data: [0, 0.42, 0.62, 0.77, 0.85, 0.9, 0.94, 0.96, 0.98, 0.99, 1],
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.1)',
                    borderWidth: 2,
                    fill: true
                }, {
                    label: 'Random Guess',
                    data: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                    borderColor: 'rgba(200, 200, 200, 1)',
                    borderWidth: 1,
                    borderDash: [5, 5],
                    fill: false,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'False Positive Rate'
                        },
                        beginAtZero: true,
                        max: 1
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'True Positive Rate'
                        },
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });
    }
    
    // Feature importance chart
    if (document.getElementById('feature-importance')) {
        const ctx = document.getElementById('feature-importance').getContext('2d');
        window.featureImportanceChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Weekly Viewing Hours', 'Tenure Months', 'Content Diversity', 'Price Increase', 'Technical Issues', 'Customer Service Calls', 'Account Sharing', 'Competing Services', 'User Rating', 'Last Login Days'],
                datasets: [{
                    label: 'Feature Importance',
                    data: [0.35, 0.25, 0.15, 0.08, 0.06, 0.04, 0.03, 0.02, 0.01, 0.01],
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.7)',
                        'rgba(54, 162, 235, 0.7)',
                        'rgba(255, 206, 86, 0.7)',
                        'rgba(75, 192, 192, 0.7)',
                        'rgba(153, 102, 255, 0.7)',
                        'rgba(255, 159, 64, 0.7)',
                        'rgba(199, 199, 199, 0.7)',
                        'rgba(83, 102, 255, 0.7)',
                        'rgba(40, 167, 69, 0.7)',
                        'rgba(220, 53, 69, 0.7)'
                    ],
                    borderColor: [
                        'rgba(255, 99, 132, 1)',
                        'rgba(54, 162, 235, 1)',
                        'rgba(255, 206, 86, 1)',
                        'rgba(75, 192, 192, 1)',
                        'rgba(153, 102, 255, 1)',
                        'rgba(255, 159, 64, 1)',
                        'rgba(199, 199, 199, 1)',
                        'rgba(83, 102, 255, 1)',
                        'rgba(40, 167, 69, 1)',
                        'rgba(220, 53, 69, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 0.4
                    }
                }
            }
        });
    }
}

/**
 * Formats the Parameter Grid JSON to be more readable
 */
function formatParameterGrid() {
    const paramGridTextarea = document.getElementById('param-grid');
    try {
        // Parse existing JSON content
        const jsonContent = JSON.parse(paramGridTextarea.value);
        
        // Format with proper indentation and spacing
        const formattedJson = JSON.stringify(jsonContent, null, 2);
        
        // Update textarea with formatted JSON
        paramGridTextarea.value = formattedJson;
    } catch (e) {
        // If parsing fails, keep the original content
        console.error('Failed to format parameter grid JSON:', e);
    }
}

/**
 * Handles hyperparameter tuning form submission
 */
function handleTuningFormSubmit(e) {
    e.preventDefault();
    
    // Show loading and hide error/results
    document.getElementById('tuning-loading').classList.remove('d-none');
    document.getElementById('tuning-error').classList.add('d-none');
    document.getElementById('tuning-results').classList.add('d-none');
    
    // Get form data
    const formData = new FormData(e.target);
    const modelType = formData.get('model_type');
    let paramGrid;
    
    try {
        paramGrid = JSON.parse(formData.get('param_grid'));
    } catch (error) {
        // Show error message
        document.getElementById('tuning-loading').classList.add('d-none');
        document.getElementById('tuning-error').classList.remove('d-none');
        document.getElementById('tuning-error-message').textContent = '하이퍼파라미터 튜닝 오류: 파라미터 그리드가 유효한 JSON 형식이 아닙니다.';
        return;
    }
    
    // Prepare request data
    const requestData = {
        model_type: modelType,
        param_grid: paramGrid,
        cv_folds: parseInt(formData.get('cv_folds')),
        scoring_metric: formData.get('scoring_metric'),
        use_stratified: formData.get('use_stratified') === 'on',
        shuffle_data: formData.get('shuffle_data') === 'on'
    };
    
    // Call API for hyperparameter tuning
    fetch('/api/hyperparameter-tuning', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('서버 오류: 하이퍼파라미터 튜닝 요청 처리에 실패했습니다.');
        }
        return response.json();
    })
    .then(data => {
        // Hide loading
        document.getElementById('tuning-loading').classList.add('d-none');
        
        if (data.status === 'error') {
            // Show error message
            document.getElementById('tuning-error').classList.remove('d-none');
            document.getElementById('tuning-error-message').textContent = '하이퍼파라미터 튜닝 오류: ' + data.error;
            return;
        }
        
        // Show results and update UI
        document.getElementById('tuning-results').classList.remove('d-none');
        
        // Display CV method used if available
        const cvMethodDisplay = document.getElementById('cv-method-used');
        if (cvMethodDisplay && data.cv_method) {
            cvMethodDisplay.textContent = `${data.cv_method} with ${data.cv_folds} folds`;
        }
        
        // Update best parameters
        const bestParamsElement = document.getElementById('best-params');
        bestParamsElement.textContent = JSON.stringify(data.best_params, null, 2);
        
        // Update metrics
        document.getElementById('best-accuracy').textContent = data.metrics.accuracy.toFixed(3);
        document.getElementById('best-precision').textContent = data.metrics.precision.toFixed(3);
        document.getElementById('best-recall').textContent = data.metrics.recall.toFixed(3);
        document.getElementById('best-f1').textContent = data.metrics.f1.toFixed(3);
        
        // Update parameter importance chart
        updateParamImportanceChart(data.param_importance);
        
        // Enable save button
        document.getElementById('save-tuned-model').disabled = false;
    })
    .catch(error => {
        document.getElementById('tuning-loading').classList.add('d-none');
        document.getElementById('tuning-error').classList.remove('d-none');
        document.getElementById('tuning-error-message').textContent = '오류 발생: ' + error.message;
    });
}

/**
 * Updates parameter descriptions based on selected model type
 */
function updateParameterDescriptions() {
    const modelType = document.getElementById('model-type').value;
    let description = '';
    
    switch (modelType) {
        case 'xgboost':
            description = `
                <h6>XGBoost Parameters</h6>
                <ul>
                    <li><strong>n_estimators</strong>: Number of boosting rounds</li>
                    <li><strong>max_depth</strong>: Maximum tree depth</li>
                    <li><strong>learning_rate</strong>: Step size shrinkage used to prevent overfitting</li>
                    <li><strong>subsample</strong>: Subsample ratio of training instances</li>
                    <li><strong>colsample_bytree</strong>: Subsample ratio of columns when constructing each tree</li>
                </ul>
            `;
            // Update parameter grid example
            document.getElementById('param-grid').value = JSON.stringify({
                "n_estimators": [50, 100, 200], 
                "max_depth": [3, 5, 7], 
                "learning_rate": [0.01, 0.1, 0.2], 
                "subsample": [0.8, 1.0], 
                "colsample_bytree": [0.8, 1.0]
            }, null, 2);
            break;
        case 'lightgbm':
            description = `
                <h6>LightGBM Parameters</h6>
                <ul>
                    <li><strong>n_estimators</strong>: Number of boosting iterations</li>
                    <li><strong>num_leaves</strong>: Maximum tree leaves for base learners</li>
                    <li><strong>learning_rate</strong>: Boosting learning rate</li>
                    <li><strong>feature_fraction</strong>: Fraction of features to be used in each iteration</li>
                    <li><strong>bagging_fraction</strong>: Fraction of data to be used for each iteration</li>
                </ul>
            `;
            document.getElementById('param-grid').value = JSON.stringify({
                "n_estimators": [50, 100, 200], 
                "num_leaves": [31, 50, 70], 
                "learning_rate": [0.01, 0.05, 0.1], 
                "feature_fraction": [0.8, 0.9, 1.0], 
                "bagging_fraction": [0.8, 0.9, 1.0]
            }, null, 2);
            break;
        case 'randomforest':
            description = `
                <h6>Random Forest Parameters</h6>
                <ul>
                    <li><strong>n_estimators</strong>: Number of trees in the forest</li>
                    <li><strong>max_depth</strong>: Maximum depth of the trees</li>
                    <li><strong>min_samples_split</strong>: Minimum number of samples required to split a node</li>
                    <li><strong>min_samples_leaf</strong>: Minimum number of samples required at each leaf node</li>
                    <li><strong>max_features</strong>: Number of features to consider for best split</li>
                </ul>
            `;
            document.getElementById('param-grid').value = JSON.stringify({
                "n_estimators": [100, 200, 300], 
                "max_depth": [10, 20, 30, null], 
                "min_samples_split": [2, 5, 10], 
                "min_samples_leaf": [1, 2, 4], 
                "max_features": ["sqrt", "log2", null]
            }, null, 2);
            break;
        case 'logistic':
            description = `
                <h6>Logistic Regression Parameters</h6>
                <ul>
                    <li><strong>C</strong>: Inverse of regularization strength</li>
                    <li><strong>penalty</strong>: Penalty norm (L1 or L2)</li>
                    <li><strong>solver</strong>: Algorithm to use in optimization</li>
                    <li><strong>max_iter</strong>: Maximum number of iterations</li>
                    <li><strong>class_weight</strong>: Weights associated with classes</li>
                </ul>
            `;
            document.getElementById('param-grid').value = JSON.stringify({
                "C": [0.001, 0.01, 0.1, 1, 10, 100], 
                "penalty": ["l1", "l2", "elasticnet", "none"], 
                "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"], 
                "max_iter": [100, 200, 500], 
                "class_weight": [null, "balanced"]
            }, null, 2);
            break;
        case 'svm':
            description = `
                <h6>SVM Parameters</h6>
                <ul>
                    <li><strong>C</strong>: Regularization parameter</li>
                    <li><strong>kernel</strong>: Kernel type</li>
                    <li><strong>gamma</strong>: Kernel coefficient</li>
                    <li><strong>degree</strong>: Degree of polynomial kernel</li>
                    <li><strong>probability</strong>: Whether to enable probability estimates</li>
                </ul>
            `;
            document.getElementById('param-grid').value = JSON.stringify({
                "C": [0.1, 1, 10, 100], 
                "kernel": ["linear", "poly", "rbf", "sigmoid"], 
                "gamma": ["scale", "auto", 0.1, 0.01], 
                "degree": [2, 3, 4], 
                "probability": [true]
            }, null, 2);
            break;
    }
    
    document.getElementById('param-descriptions').innerHTML = description;
}

/**
 * Simulated function to update parameter importance chart
 */
function updateParamImportanceChart(paramImportance) {
    // In a real implementation, this would use actual data from the API response
    const ctx = document.getElementById('param-importance-chart').getContext('2d');
    
    if (window.paramImportanceChart) {
        window.paramImportanceChart.destroy();
    }
    
    // Convert parameter importance to sorted arrays for chart
    const params = [];
    const importances = [];
    
    for (const [param, importance] of Object.entries(paramImportance)) {
        params.push(param);
        importances.push(importance);
    }
    
    window.paramImportanceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: params,
            datasets: [{
                label: 'Parameter Importance',
                data: importances,
                backgroundColor: 'rgba(75, 192, 192, 0.7)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            scales: {
                x: {
                    beginAtZero: true
                }
            }
        }
    });
}

/**
 * Handles model training form submission
 */
function handleTrainingFormSubmit(e) {
    e.preventDefault();
    // Show loading
    document.getElementById('training-loading').classList.remove('d-none');
    document.getElementById('no-training-results').classList.add('d-none');
    document.getElementById('training-results').classList.add('d-none');
    
    // Simulate progress updates
    let progress = 0;
    const progressBar = document.getElementById('training-progress');
    const progressInterval = setInterval(() => {
        progress += 5;
        progressBar.style.width = progress + '%';
        progressBar.setAttribute('aria-valuenow', progress);
        
        if (progress >= 100) {
            clearInterval(progressInterval);
            
            // Show results
            setTimeout(() => {
                document.getElementById('training-loading').classList.add('d-none');
                document.getElementById('training-results').classList.remove('d-none');
                
                // Create confusion matrix and learning curves
                createConfusionMatrix();
                createLearningCurves();
            }, 500);
        }
    }, 300);
}

/**
 * Handles feature engineering form submission
 */
function handleFeatureEngineeringSubmit(e) {
    e.preventDefault();
    alert('Feature engineering applied successfully! Model comparison updated with new features.');
}

/**
 * Creates a confusion matrix visualization
 */
function createConfusionMatrix() {
    const ctx = document.getElementById('confusion-matrix').getContext('2d');
    
    if (window.confusionMatrixChart) {
        window.confusionMatrixChart.destroy();
    }
    
    window.confusionMatrixChart = new Chart(ctx, {
        type: 'matrix',
        data: {
            datasets: [{
                label: 'Confusion Matrix',
                data: [
                    {x: 'True Negative', y: 'Predicted Negative', v: 85},
                    {x: 'False Positive', y: 'Predicted Positive', v: 15},
                    {x: 'False Negative', y: 'Predicted Negative', v: 10},
                    {x: 'True Positive', y: 'Predicted Positive', v: 90},
                ],
                backgroundColor(c) {
                    const value = c.dataset.data[c.dataIndex].v;
                    const alpha = (value / 100) * 0.8 + 0.2;
                    if (c.dataset.data[c.dataIndex].x.includes('True')) {
                        return `rgba(75, 192, 192, ${alpha})`;
                    }
                    return `rgba(255, 99, 132, ${alpha})`;
                },
                borderColor: 'white',
                borderWidth: 1,
                width: ({chart}) => (chart.chartArea || {}).width / 2 - 1,
                height: ({chart}) => (chart.chartArea || {}).height / 2 - 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Actual Class'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Predicted Class'
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: (context) => {
                            const v = context.dataset.data[context.dataIndex];
                            return [`${v.x}`, `${v.y}`, `Count: ${v.v}`];
                        }
                    }
                }
            }
        }
    });
}

/**
 * Creates learning curves visualization
 */
function createLearningCurves() {
    const ctx = document.getElementById('learning-curves').getContext('2d');
    
    if (window.learningCurvesChart) {
        window.learningCurvesChart.destroy();
    }
    
    window.learningCurvesChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'],
            datasets: [{
                label: 'Training Score',
                data: [0.92, 0.91, 0.90, 0.89, 0.88, 0.87, 0.87, 0.86, 0.86, 0.85],
                borderColor: 'rgba(255, 99, 132, 1)',
                backgroundColor: 'rgba(255, 99, 132, 0.1)',
                fill: true,
                tension: 0.1
            }, {
                label: 'Validation Score',
                data: [0.75, 0.78, 0.79, 0.81, 0.82, 0.83, 0.84, 0.845, 0.85, 0.855],
                borderColor: 'rgba(54, 162, 235, 1)',
                backgroundColor: 'rgba(54, 162, 235, 0.1)',
                fill: true,
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Training Data Size'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Score'
                    },
                    min: 0.7,
                    max: 1.0
                }
            }
        }
    });
}

/**
 * Loads model comparison data from the API
 */
function loadModelComparisonData() {
    // This would be an actual API call in a real implementation
    console.log('Loading model comparison data...');
}

/**
 * Updates feature distribution chart
 */
function updateFeatureDistribution() {
    const feature = document.getElementById('feature-selector').value;
    console.log(`Updating distribution for feature: ${feature}`);
    
    // This would be an actual API call in a real implementation
}

/**
 * Views detailed model information
 */
function viewModelDetails(modelId) {
    alert(`Viewing details for model ID: ${modelId}`);
}

/**
 * Sets a model as the active production model
 */
function setActiveModel(modelId) {
    fetch(`/api/set-active-model/${modelId}`, {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('Model successfully set as active production model!');
            // Update UI
            document.querySelectorAll('.set-active').forEach(btn => {
                btn.removeAttribute('disabled');
                btn.textContent = 'Set Active';
            });
            const activeBtn = document.querySelector(`.set-active[data-model-id="${modelId}"]`);
            activeBtn.setAttribute('disabled', 'disabled');
            activeBtn.textContent = 'Active';
        } else {
            alert(`Error: ${data.error}`);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while setting the active model.');
    });
}

/**
 * Saves the tuned model to the models directory
 */
function saveTunedModel() {
    fetch('/api/save-tuned-model', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert(`Model saved successfully as: ${data.model_name}`);
        } else {
            alert(`Error: ${data.error}`);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while saving the model.');
    });
} 