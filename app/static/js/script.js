async function analyzeRisk() {
    const input = document.getElementById('tickerInput');
    const button = document.getElementById('analyzeBtn');
    const btnText = button.querySelector('span');
    const loader = document.getElementById('loader');
    const resultDiv = document.getElementById('resultContainer');
    const errorDiv = document.getElementById('errorContainer');
    const ticker = input.value.trim().toUpperCase();

    if (!ticker) return;

    // Reset UI
    button.disabled = true;
    btnText.style.display = 'none';
    loader.style.display = 'block';
    resultDiv.classList.add('hidden');
    errorDiv.classList.add('hidden');

    try {
        const response = await fetch('/predict_risk', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker: ticker })
        });

        const data = await response.json();

        if (!response.ok) throw new Error(data.detail || 'Prediction failed');

        // Populate Data
        document.getElementById('tickerLabel').textContent = ticker;
        
        // Risk Badge
        const badge = document.getElementById('riskBadge');
        badge.className = 'badge';
        if (data.risk_class === 'Low') badge.classList.add('low');
        else if (data.risk_class === 'Medium') badge.classList.add('med');
        else badge.classList.add('high');
        badge.textContent = `${data.risk_class.toUpperCase()} RISK`;

        document.getElementById('volatilityVal').textContent = (data.volatility * 100).toFixed(2) + '%';
        document.getElementById('confidenceVal').textContent = (data.confidence_score * 100).toFixed(0) + '%';
        document.getElementById('recommendationVal').textContent = data.recommendation;

        // Probabilities
        updateBar('probLow', 'percLow', data.probabilities.Low);
        updateBar('probMed', 'percMed', data.probabilities.Medium);
        updateBar('probHigh', 'percHigh', data.probabilities.High);

        resultDiv.classList.remove('hidden');

    } catch (err) {
        document.getElementById('errorMsg').textContent = err.message;
        errorDiv.classList.remove('hidden');
    } finally {
        button.disabled = false;
        btnText.style.display = 'block';
        loader.style.display = 'none';
    }
}

function updateBar(barId, textId, value) {
    const perc = (value * 100).toFixed(1) + '%';
    document.getElementById(barId).style.width = perc;
    document.getElementById(textId).textContent = perc;
}

// Enter key support
document.getElementById('tickerInput').addEventListener('keypress', function (e) {
    if (e.key === 'Enter') analyzeRisk();
});
