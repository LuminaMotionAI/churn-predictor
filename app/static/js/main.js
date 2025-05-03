/**
 * Netflix Churn Predictor - Main JavaScript
 * Common functionality for the Netflix Churn Prediction System dashboard
 */

// Initialize tooltips and popovers
document.addEventListener('DOMContentLoaded', function() {
    // Initialize any Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize any Bootstrap popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function(popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // Set active nav item based on current page
    setActiveNavItem();
    
    // Initialize dashboard real-time updates if on dashboard page
    if (document.getElementById('subscription-filter')) {
        initDashboardFilters();
    }
});

/**
 * Sets the active navigation item based on the current URL
 */
function setActiveNavItem() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
    
    navLinks.forEach(link => {
        link.classList.remove('active');
        const href = link.getAttribute('href');
        
        if (currentPath === href || 
            (href !== '/' && currentPath.startsWith(href))) {
            link.classList.add('active');
        } else if (currentPath === '/' && href === '/') {
            link.classList.add('active');
        }
    });
}

/**
 * Initializes dashboard filters functionality
 */
function initDashboardFilters() {
    // Set up apply filters button
    const applyFiltersBtn = document.getElementById('apply-filters');
    if (applyFiltersBtn) {
        applyFiltersBtn.addEventListener('click', applyDashboardFilters);
    }
    
    // Set up reset filters button
    const resetFiltersBtn = document.getElementById('reset-filters');
    if (resetFiltersBtn) {
        resetFiltersBtn.addEventListener('click', resetDashboardFilters);
    }
    
    // Initialize filter state to apply any stored filter settings
    loadSavedFilterSettings();
    
    // Check if we have any saved filters, and if so, apply them
    const hasFilters = sessionStorage.getItem('filter_subscription') || 
                      sessionStorage.getItem('filter_tenure') || 
                      sessionStorage.getItem('filter_viewing') || 
                      sessionStorage.getItem('filter_risk');
    
    if (hasFilters) {
        // Apply saved filters automatically
        applyDashboardFilters();
    }
}

/**
 * Applies dashboard filters
 */
function applyDashboardFilters() {
    const subscriptionType = document.getElementById('subscription-filter').value;
    const tenure = document.getElementById('tenure-filter').value;
    const viewingHours = document.getElementById('viewing-filter').value;
    const riskLevel = document.getElementById('risk-filter').value;
    
    // Save filters to sessionStorage
    sessionStorage.setItem('filter_subscription', subscriptionType);
    sessionStorage.setItem('filter_tenure', tenure);
    sessionStorage.setItem('filter_viewing', viewingHours);
    sessionStorage.setItem('filter_risk', riskLevel);
    
    // Fetch filtered data from the server
    fetchFilteredDashboardData({
        subscription_type: subscriptionType,
        tenure: tenure,
        viewing_hours: viewingHours,
        risk_level: riskLevel
    });
    
    // Show loading indicator
    showFilteringLoadingState();
}

/**
 * Resets dashboard filters to defaults
 */
function resetDashboardFilters() {
    // Reset filter dropdowns
    document.getElementById('subscription-filter').value = 'all';
    document.getElementById('tenure-filter').value = 'all';
    document.getElementById('viewing-filter').value = 'all';
    document.getElementById('risk-filter').value = 'all';
    
    // Clear saved filters
    sessionStorage.removeItem('filter_subscription');
    sessionStorage.removeItem('filter_tenure');
    sessionStorage.removeItem('filter_viewing');
    sessionStorage.removeItem('filter_risk');
    
    // Fetch unfiltered data
    fetchFilteredDashboardData({
        subscription_type: 'all',
        tenure: 'all',
        viewing_hours: 'all',
        risk_level: 'all'
    });
    
    // Show loading indicator
    showFilteringLoadingState();
}

/**
 * Shows loading state while filter is being applied
 */
function showFilteringLoadingState() {
    // Add loading indicator to charts
    const chartContainers = document.querySelectorAll('.card-body canvas');
    chartContainers.forEach(container => {
        const parent = container.parentNode;
        const loadingOverlay = document.createElement('div');
        loadingOverlay.className = 'filtering-overlay';
        loadingOverlay.innerHTML = '<div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div>';
        parent.appendChild(loadingOverlay);
    });
}

/**
 * Loads any saved filter settings from previous session
 */
function loadSavedFilterSettings() {
    const subscriptionFilter = sessionStorage.getItem('filter_subscription');
    const tenureFilter = sessionStorage.getItem('filter_tenure');
    const viewingFilter = sessionStorage.getItem('filter_viewing');
    const riskFilter = sessionStorage.getItem('filter_risk');
    
    if (subscriptionFilter) {
        document.getElementById('subscription-filter').value = subscriptionFilter;
    }
    
    if (tenureFilter) {
        document.getElementById('tenure-filter').value = tenureFilter;
    }
    
    if (viewingFilter) {
        document.getElementById('viewing-filter').value = viewingFilter;
    }
    
    if (riskFilter) {
        document.getElementById('risk-filter').value = riskFilter;
    }
}

/**
 * Fetches filtered dashboard data based on selected filters
 */
function fetchFilteredDashboardData(filters) {
    // In a real implementation, this would use query parameters
    fetch('/api/churn-data?' + new URLSearchParams(filters).toString())
        .then(response => response.json())
        .then(data => {
            updateDashboardWithFilteredData(data);
        })
        .catch(error => {
            console.error('Error fetching filtered data:', error);
            // Remove loading overlays
            removeFilteringLoadingState();
            
            // Show error message
            const dashboardContainer = document.querySelector('.container-fluid');
            const errorAlert = document.createElement('div');
            errorAlert.className = 'alert alert-danger alert-dismissible fade show mt-3';
            errorAlert.innerHTML = `
                <strong>필터링 오류:</strong> 데이터를 불러오는 중 문제가 발생했습니다.
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            `;
            dashboardContainer.prepend(errorAlert);
        });
}

/**
 * Removes loading state after filter is applied
 */
function removeFilteringLoadingState() {
    const loadingOverlays = document.querySelectorAll('.filtering-overlay');
    loadingOverlays.forEach(overlay => {
        overlay.remove();
    });
}

/**
 * Updates dashboard with filtered data
 */
function updateDashboardWithFilteredData(data) {
    // Remove loading overlays
    removeFilteringLoadingState();
    
    // In a real implementation, this would update all dashboard elements
    // For now, we'll just update the key metrics
    if (document.getElementById('total-users')) {
        document.getElementById('total-users').textContent = data.total_users;
    }
    
    if (document.getElementById('at-risk-users')) {
        document.getElementById('at-risk-users').textContent = data.at_risk_count;
    }
    
    if (document.getElementById('churn-rate')) {
        document.getElementById('churn-rate').textContent = data.at_risk_percentage + '%';
    }
    
    // Add more chart updates here
    
    // Show success message
    const dashboardContainer = document.querySelector('.container-fluid');
    const successAlert = document.createElement('div');
    successAlert.className = 'alert alert-success alert-dismissible fade show mt-3';
    successAlert.innerHTML = `
        <strong>필터 적용 완료:</strong> 대시보드가 업데이트되었습니다.
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    dashboardContainer.prepend(successAlert);
    
    // Auto-remove the alert after 3 seconds
    setTimeout(() => {
        successAlert.remove();
    }, 3000);
}

/**
 * Initializes real-time dashboard updates
 */
function initDashboardUpdates() {
    // Simulate periodic data updates
    setInterval(fetchDashboardData, 60000); // Update every minute
}

/**
 * Fetches updated dashboard data via API
 */
function fetchDashboardData() {
    fetch('/api/churn-data')
        .then(response => response.json())
        .then(data => {
            updateDashboardMetrics(data);
        })
        .catch(error => {
            console.error('Error fetching dashboard data:', error);
        });
}

/**
 * Updates dashboard metrics with new data
 */
function updateDashboardMetrics(data) {
    // Update counter elements if they exist
    if (document.getElementById('total-users')) {
        document.getElementById('total-users').textContent = data.total_users;
    }
    
    if (document.getElementById('at-risk-users')) {
        document.getElementById('at-risk-users').textContent = data.at_risk_count;
    }
    
    if (document.getElementById('churn-rate')) {
        document.getElementById('churn-rate').textContent = data.at_risk_percentage + '%';
    }
    
    // Add a subtle animation to show the update
    const metrics = document.querySelectorAll('.metric-value');
    metrics.forEach(metric => {
        metric.classList.add('metric-updated');
        setTimeout(() => {
            metric.classList.remove('metric-updated');
        }, 1000);
    });
}

/**
 * Formats a number with commas as thousands separators
 */
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

/**
 * Handles API call error responses
 */
function handleApiError(error, elementId) {
    console.error('API Error:', error);
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = `<div class="alert alert-danger">
            An error occurred while fetching data. Please try again later.
        </div>`;
    }
}

/**
 * Creates a SHAP waterfall chart for a specific user
 */
function createShapWaterfallChart(containerId, data) {
    if (!data || !data.features || !data.values) {
        console.error('Invalid SHAP data provided');
        return;
    }
    
    const ctx = document.getElementById(containerId).getContext('2d');
    const chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.features,
            datasets: [{
                label: 'SHAP Value',
                data: data.values,
                backgroundColor: data.values.map(value => 
                    value > 0 ? 'rgba(255, 99, 132, 0.7)' : 'rgba(75, 192, 192, 0.7)'
                ),
                borderColor: data.values.map(value => 
                    value > 0 ? 'rgba(255, 99, 132, 1)' : 'rgba(75, 192, 192, 1)'
                ),
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.raw;
                            const sign = value >= 0 ? '+' : '';
                            return `Impact: ${sign}${value.toFixed(4)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: function(context) {
                            if (context.tick.value === 0) {
                                return 'rgba(0, 0, 0, 0.5)';
                            }
                            return 'rgba(0, 0, 0, 0.1)';
                        },
                        lineWidth: function(context) {
                            if (context.tick.value === 0) {
                                return 2;
                            }
                            return 1;
                        }
                    }
                }
            }
        }
    });
}

/**
 * Exports dashboard as PDF
 */
function exportDashboardAsPdf() {
    alert('Exporting dashboard as PDF... This feature will be implemented soon.');
}

/**
 * Makes an API call to the churn prediction endpoint
 */
function predictChurn(userData) {
    return fetch('/api/predict-churn', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(userData),
    })
    .then(response => response.json());
}

// Export functions for use in other scripts
window.churnPredictor = {
    formatNumber,
    handleApiError,
    createShapWaterfallChart,
    exportDashboardAsPdf,
    predictChurn,
    applyDashboardFilters,
    resetDashboardFilters
}; 