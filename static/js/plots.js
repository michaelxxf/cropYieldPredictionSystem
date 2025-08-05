// Function to fetch historical data
function fetchHistoricalData() {
    fetch('/get_historical_predictions')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Error fetching historical data:', data.error);
                return;
            }
            
            if (data.yield_trend && data.parameter_impact) {
                initializePlots(
                    JSON.parse(data.yield_trend),
                    JSON.parse(data.parameter_impact)
                );
            } else {
                console.log('No historical data available');
                // Show empty state message
                document.getElementById('yieldTrendPlot').innerHTML = '<div class="text-center py-5"><i class="bi bi-bar-chart-line display-4 text-muted"></i><h4 class="mt-3 text-muted">No Historical Data Available</h4><p class="text-muted">Make some predictions to see historical analysis and trends.</p></div>';
                document.getElementById('parameterImpactPlot').innerHTML = '<div class="text-center py-5"><i class="bi bi-bar-chart-line display-4 text-muted"></i><h4 class="mt-3 text-muted">No Historical Data Available</h4><p class="text-muted">Make some predictions to see historical analysis and trends.</p></div>';
            }
        })
        .catch(error => {
            console.error('Error fetching historical data:', error);
        });
}

function initializePlots(yieldTrendData, parameterImpactData) {
    if (yieldTrendData) {
        try {
            console.log("Initializing yield trend plot");
            Plotly.newPlot('yieldTrendPlot', yieldTrendData.data, yieldTrendData.layout, {
                responsive: true,
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['lasso2d', 'select2d']
            })
            .then(() => console.log("Yield trend plot created successfully"))
            .catch(error => console.error("Error creating yield trend plot:", error));
        } catch (error) {
            console.error("Error processing yield trend data:", error);
        }
    } else {
        console.log("No yield trend data available");
    }

    if (parameterImpactData) {
        try {
            console.log("Initializing parameter impact plot");
            Plotly.newPlot('parameterImpactPlot', parameterImpactData.data, parameterImpactData.layout, {
                responsive: true,
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['lasso2d', 'select2d']
            })
            .then(() => console.log("Parameter impact plot created successfully"))
            .catch(error => console.error("Error creating parameter impact plot:", error));
        } catch (error) {
            console.error("Error processing parameter impact data:", error);
        }
    }
}

// Initialize plots when the page loads
document.addEventListener('DOMContentLoaded', function() {
    fetchHistoricalData();
    
    // Refresh historical data every 5 minutes
    setInterval(fetchHistoricalData, 5 * 60 * 1000);
}); 