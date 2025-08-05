// Form submission handler
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    if (!form) return;

    form.addEventListener('submit', function(event) {
        event.preventDefault();
        
        if (!this.checkValidity()) {
            event.stopPropagation();
            this.classList.add('was-validated');
            return;
        }

        const formData = {
            nitrogen: parseFloat(document.getElementById('nitrogen').value),
            phosphorus: parseFloat(document.getElementById('phosphorus').value),
            potassium: parseFloat(document.getElementById('potassium').value),
            temperature: parseFloat(document.getElementById('temperature').value),
            humidity: parseFloat(document.getElementById('humidity').value),
            ph: parseFloat(document.getElementById('ph').value),
            rainfall: parseFloat(document.getElementById('rainfall').value)
        };

        // Show loading state
        const submitBtn = this.querySelector('button[type="submit"]');
        const originalText = submitBtn.innerHTML;
        submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Predicting...';
        submitBtn.disabled = true;

        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        })        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            
            console.log('Prediction response:', data); // Log the response
            
            // Create prediction result HTML
            const resultHtml = `
                <div class="grid-item" style="grid-column: 1 / -1;">
                    <div class="prediction-result">
                        <div class="card border-success">
                            <div class="card-header bg-success text-white">
                                <h5 class="card-title mb-0">
                                    <i class="bi bi-check-circle"></i> Prediction Results
                                </h5>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-md-6">
                                        <h6 class="card-subtitle mb-2 text-muted">Predicted Yield</h6>                                        <p class="display-6 text-success">${data.yield_prediction} tons</p>
                                        ${data.confidence ? 
                                            `<div class="progress mb-3">
                                                <div class="progress-bar bg-success" role="progressbar" 
                                                    style="width: ${data.confidence}%" 
                                                    aria-valuenow="${data.confidence}" 
                                                    aria-valuemin="0" 
                                                    aria-valuemax="100">
                                                    ${data.confidence}% Confidence
                                                </div>
                                            </div>` : ''
                                        }
                                    </div>
                                    <div class="col-md-6">
                                        <h6 class="card-subtitle mb-2 text-muted">Recommendations</h6>
                                        <p class="card-text">${data.recommendations.split('\n').join('<br>')}</p>
                                    </div>
                                    ${data.stats ? 
                                        `<div class="col-12 mt-3">
                                            <h6 class="card-subtitle mb-2 text-muted">Historical Statistics</h6>
                                            <div class="row">
                                                <div class="col-4">
                                                    <small class="text-muted">Average Yield:</small>
                                                    <p class="mb-2">${data.stats.average} tons</p>
                                                </div>
                                                <div class="col-4">
                                                    <small class="text-muted">Maximum:</small>
                                                    <p class="mb-2">${data.stats.maximum} tons</p>
                                                </div>
                                                <div class="col-4">
                                                    <small class="text-muted">Minimum:</small>
                                                    <p class="mb-2">${data.stats.minimum} tons</p>
                                                </div>
                                            </div>
                                        </div>` : ''
                                    }
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;

            // Find the prediction form card
            const formCard = document.querySelector('#predictionForm').closest('.grid-item');
            
            // Remove any existing prediction results
            const existingResult = document.querySelector('.prediction-result');
            if (existingResult) {
                existingResult.closest('.grid-item').remove();
            }
            
            // Insert new result after the form
            formCard.insertAdjacentHTML('afterend', resultHtml);

            // Smooth scroll to the result
            document.querySelector('.prediction-result').scrollIntoView({ behavior: 'smooth' });
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error making prediction: ' + error.message);
        })
        .finally(() => {
            submitBtn.innerHTML = originalText;
            submitBtn.disabled = false;
        });
    });
});
