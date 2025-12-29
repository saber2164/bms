document.addEventListener('DOMContentLoaded', () => {
    const predictBtn = document.getElementById('predict-btn');
    const socDisplay = document.getElementById('soc-display');
    const ctx = document.getElementById('socChart').getContext('2d');

    // State to store history of inputs
    let history = [];

    // Initialize Chart
    let chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Estimated SoC (%)',
                data: [],
                borderColor: '#ffffff', // White
                backgroundColor: 'rgba(255, 255, 255, 0.1)',
                borderWidth: 2,
                pointBackgroundColor: '#ffffff',
                pointRadius: 3,
                tension: 0.1, // Less smooth, more technical
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: {
                        color: '#333333'
                    },
                    ticks: {
                        color: '#ffffff',
                        font: {
                            family: 'Courier New'
                        }
                    },
                    title: { display: true, text: 'State of Charge (%)', color: '#ffffff', font: { family: 'Courier New', weight: 'bold' } }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: '#ffffff',
                        font: {
                            family: 'Courier New'
                        }
                    },
                    title: { display: true, text: 'Time Step', color: '#ffffff', font: { family: 'Courier New', weight: 'bold' } }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#ffffff',
                        font: {
                            family: 'Courier New'
                        }
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: '#ffffff',
                    titleColor: '#000000',
                    bodyColor: '#000000',
                    borderColor: '#ffffff',
                    borderWidth: 1,
                    titleFont: { family: 'Courier New' },
                    bodyFont: { family: 'Courier New' }
                }
            }
        }
    });

    predictBtn.addEventListener('click', async () => {
        // Get inputs
        const voltage = parseFloat(document.getElementById('voltage').value);
        const current = parseFloat(document.getElementById('current').value);
        const temperature = parseFloat(document.getElementById('temperature').value);

        if (isNaN(voltage) || isNaN(current) || isNaN(temperature)) {
            alert("Please enter valid numbers for all fields.");
            return;
        }

        // Call API with single measurement (not history array)
        try {
            const response = await fetch('/api/predict_soc', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    voltage: voltage,
                    current: current,
                    temperature: temperature
                })
            });

            const data = await response.json();

            if (data.status === 'success') {
                const latestSoC = (data.soc * 100).toFixed(1); // Convert to percentage

                // Update Display
                socDisplay.innerText = `${latestSoC} %`;

                // Add to history for chart
                history.push({ soc: data.soc, q_max: data.q_max, r0: data.r0 });

                // Update Chart
                const labels = history.map((_, index) => index + 1);
                const socValues = history.map(h => h.soc * 100);

                chart.data.labels = labels;
                chart.data.datasets[0].data = socValues;
                chart.update();

            } else {
                alert(`Error: ${data.message}`);
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to get prediction: ' + error.message);
        }
    });
});
