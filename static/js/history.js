// Global variables
let currentData = null;
let currentTimeRange = 30;

// Initialize the page
document.addEventListener('DOMContentLoaded', function() {
    // Set up event listeners
    document.getElementById('timeRange').addEventListener('change', handleTimeRangeChange);
    document.getElementById('exportData').addEventListener('click', exportData);
    
    // Initial data load
    fetchHistoricalData(currentTimeRange);
});

// Handle time range changes
function handleTimeRangeChange(event) {
    currentTimeRange = parseInt(event.target.value);
    fetchHistoricalData(currentTimeRange);
}

// Fetch historical data
function fetchHistoricalData(days) {
    // Show loading state
    showLoading();
    
    fetch(`/get_historical_predictions?days=${days}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showError(data.error);
                return;
            }
            
            currentData = data;
            updateDashboard(data);
        })
        .catch(error => {
            console.error('Error fetching historical data:', error);
            showError('Failed to fetch historical data');
        })
        .finally(() => {
            hideLoading();
        });
}

// Update all dashboard components
function updateDashboard(data) {
    updateStatistics(data.predictions);
    updateYieldTrend(data.yield_trend);
    updateParameterImpact(data.parameter_impact);
    updateCorrelationMatrix(data.predictions);
    updateHistoryTable(data.predictions);
}

// Update statistics
function updateStatistics(predictions) {
    if (!predictions || predictions.length === 0) {
        document.getElementById('avgYield').textContent = '-';
        document.getElementById('maxYield').textContent = '-';
        document.getElementById('minYield').textContent = '-';
        document.getElementById('totalPredictions').textContent = '0';
        return;
    }
    
    const yields = predictions.map(p => p.yield_prediction);
    const avg = yields.reduce((a, b) => a + b, 0) / yields.length;
    const max = Math.max(...yields);
    const min = Math.min(...yields);
    
    document.getElementById('avgYield').textContent = avg.toFixed(2) + ' tons';
    document.getElementById('maxYield').textContent = max.toFixed(2) + ' tons';
    document.getElementById('minYield').textContent = min.toFixed(2) + ' tons';
    document.getElementById('totalPredictions').textContent = predictions.length;
}

// Update yield trend plot
function updateYieldTrend(yieldTrendData) {
    if (!yieldTrendData) {
        document.getElementById('yieldTrendPlot').innerHTML = '<div class="text-center py-5">No yield trend data available</div>';
        return;
    }
    
    const data = JSON.parse(yieldTrendData);
    Plotly.newPlot('yieldTrendPlot', data.data, data.layout, {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d']
    });
}

// Update parameter impact plot
function updateParameterImpact(parameterImpactData) {
    if (!parameterImpactData) {
        document.getElementById('parameterImpactPlot').innerHTML = '<div class="text-center py-5">No parameter impact data available</div>';
        return;
    }
    
    const data = JSON.parse(parameterImpactData);
    Plotly.newPlot('parameterImpactPlot', data.data, data.layout, {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d']
    });
}

// Update correlation matrix
function updateCorrelationMatrix(predictions) {
    if (!predictions || predictions.length === 0) {
        document.getElementById('correlationMatrix').innerHTML = '<div class="text-center py-5">No data available for correlation analysis</div>';
        return;
    }
    
    const parameters = ['nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall', 'yield_prediction'];
    const correlationData = [];
    
    // Calculate correlation matrix
    for (let i = 0; i < parameters.length; i++) {
        const row = [];
        for (let j = 0; j < parameters.length; j++) {
            const corr = calculateCorrelation(
                predictions.map(p => p[parameters[i]]),
                predictions.map(p => p[parameters[j]])
            );
            row.push(corr);
        }
        correlationData.push(row);
    }
    
    // Create heatmap
    const data = [{
        z: correlationData,
        x: parameters.map(p => p.charAt(0).toUpperCase() + p.slice(1)),
        y: parameters.map(p => p.charAt(0).toUpperCase() + p.slice(1)),
        type: 'heatmap',
        colorscale: 'RdBu',
        zmin: -1,
        zmax: 1
    }];
    
    const layout = {
        title: 'Parameter Correlations',
        height: 300,
        margin: { t: 50, b: 50, l: 50, r: 50 }
    };
    
    Plotly.newPlot('correlationMatrix', data, layout, {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    });
}

// Update history table
function updateHistoryTable(predictions) {
    const tbody = document.querySelector('#historyTable tbody');
    tbody.innerHTML = '';
    
    if (!predictions || predictions.length === 0) {
        tbody.innerHTML = '<tr><td colspan="10" class="text-center">No historical data available</td></tr>';
        return;
    }
    
    predictions.forEach(prediction => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${new Date(prediction.created_at).toLocaleString()}</td>
            <td>${prediction.nitrogen}</td>
            <td>${prediction.phosphorus}</td>
            <td>${prediction.potassium}</td>
            <td>${prediction.temperature}°C</td>
            <td>${prediction.humidity}%</td>
            <td>${prediction.ph}</td>
            <td>${prediction.rainfall}mm</td>
            <td>${prediction.yield_prediction.toFixed(2)} tons</td>
            <td>
                <div class="action-buttons">
                    <button class="btn btn-sm btn-outline-primary" onclick="viewDetails(${prediction.id})">
                        <i class="bi bi-eye"></i>
                    </button>
                    <button class="btn btn-sm btn-outline-danger" onclick="deletePrediction(${prediction.id})">
                        <i class="bi bi-trash"></i>
                    </button>
                </div>
            </td>
        `;
        tbody.appendChild(row);
    });
}

// Export data to CSV
function exportData() {
    if (!currentData || !currentData.predictions) {
        alert('No data available to export');
        return;
    }
    
    const predictions = currentData.predictions;
    const headers = ['Date', 'N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall', 'Predicted Yield'];
    const csvContent = [
        headers.join(','),
        ...predictions.map(p => [
            new Date(p.created_at).toLocaleString(),
            p.nitrogen,
            p.phosphorus,
            p.potassium,
            p.temperature,
            p.humidity,
            p.ph,
            p.rainfall,
            p.yield_prediction
        ].join(','))
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `crop_predictions_${new Date().toISOString().split('T')[0]}.csv`;
    link.click();
}

// Helper function to calculate correlation
function calculateCorrelation(x, y) {
    const n = x.length;
    if (n !== y.length) return 0;
    
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((a, b, i) => a + b * y[i], 0);
    const sumX2 = x.reduce((a, b) => a + b * b, 0);
    const sumY2 = y.reduce((a, b) => a + b * b, 0);
    
    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    
    return denominator === 0 ? 0 : numerator / denominator;
}

// Show loading state
function showLoading() {
    document.querySelectorAll('.card-body').forEach(el => {
        el.classList.add('loading');
    });
}

// Hide loading state
function hideLoading() {
    document.querySelectorAll('.card-body').forEach(el => {
        el.classList.remove('loading');
    });
}

// Show error message
function showError(message) {
    alert(message);
}

// View prediction details
function viewDetails(id) {
    // TODO: Implement detailed view modal
    console.log('View details for prediction:', id);
}

// Delete prediction
function deletePrediction(id) {
    if (confirm('Are you sure you want to delete this prediction?')) {
        fetch(`/delete_prediction/${id}`, {
            method: 'DELETE'
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showError(data.error);
            } else {
                fetchHistoricalData(currentTimeRange);
            }
        })
        .catch(error => {
            console.error('Error deleting prediction:', error);
            showError('Failed to delete prediction');
        });
    }
} 