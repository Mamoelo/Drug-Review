/**
 * Drug Advisor - Result Page Chart Initialization
 */

document.addEventListener('DOMContentLoaded', function() {
    initProbabilityChart();
});

/**
 * Initialize the probability chart on result page
 */
function initProbabilityChart() {
    const canvas = document.getElementById('probabilityChart');
    if (!canvas) return;
    
    // Get probability data from data attribute
    const probDataElement = document.getElementById('probability-data');
    if (!probDataElement) return;
    
    try {
        const probData = JSON.parse(probDataElement.textContent);
        createChart(canvas, probData);
    } catch (error) {
        console.error('Error parsing probability data:', error);
    }
}

/**
 * Create Chart.js bar chart
 * @param {HTMLCanvasElement} canvas - Canvas element
 * @param {Object} probData - Probability data object
 */
function createChart(canvas, probData) {
    const ctx = canvas.getContext('2d');
    
    const labels = Object.keys(probData);
    const values = Object.values(probData).map(v => parseFloat((v * 100).toFixed(1)));
    
    const colors = ['#0d6efd', '#198754', '#dc3545', '#fd7e14', '#6f42c1'];
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Probability (%)',
                data: values,
                backgroundColor: colors.slice(0, labels.length),
                borderRadius: 5,
                borderSkipped: false,
                barPercentage: 0.7,
                categoryPercentage: 0.8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Probability: ${context.parsed.y.toFixed(1)}%`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Probability (%)',
                        font: {
                            weight: 'bold',
                            size: 12
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    },
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        font: {
                            size: 11
                        }
                    }
                }
            },
            layout: {
                padding: {
                    top: 10,
                    bottom: 10
                }
            }
        }
    });
}

/**
 * Update chart with new data (if needed for dynamic updates)
 * @param {Object} newData - New probability data
 */
function updateChart(newData) {
    const chart = Chart.getChart('probabilityChart');
    if (!chart) return;
    
    const labels = Object.keys(newData);
    const values = Object.values(newData).map(v => v * 100);
    
    chart.data.labels = labels;
    chart.data.datasets[0].data = values;
    chart.update();
}