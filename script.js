// Global variables
let dataset = null;
let model = null;
let featureColumns = [];
let targetColumn = '';
let modelType = 'classification';
let trainedStats = null;
let featureImportances = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
});

function initializeEventListeners() {
    // File upload
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');

    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    fileInput.addEventListener('change', handleFileSelect);

    // Model configuration
    document.getElementById('targetColumn').addEventListener('change', updateTargetColumn);
    document.getElementById('modelType').addEventListener('change', updateModelType);
    document.getElementById('trainSize').addEventListener('input', updateTrainSize);
    document.getElementById('trainModel').addEventListener('click', trainModel);
    document.getElementById('makePrediction').addEventListener('click', makePrediction);
}

function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('dragover');
}

function handleDragLeave(e) {
    e.currentTarget.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        processFile(file);
    }
}

function processFile(file) {
    if (!file.name.endsWith('.csv')) {
        showStatus('Please upload a CSV file.', 'error');
        return;
    }

    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            Papa.parse(e.target.result, {
                header: true,
                dynamicTyping: true,
                skipEmptyLines: true,
                complete: function(results) {
                    dataset = results.data;
                    displayDataPreview();
                    showModelConfig();
                },
                error: function(error) {
                    showStatus('Error parsing CSV: ' + error.message, 'error');
                }
            });
        } catch (error) {
            showStatus('Error reading file: ' + error.message, 'error');
        }
    };
    reader.readAsText(file);
}

function displayDataPreview() {
    const previewDiv = document.getElementById('dataPreview');
    const statsDiv = document.getElementById('dataStats');
    const tableDiv = document.getElementById('dataTable');

    // Show statistics
    statsDiv.innerHTML = `📊 Dataset loaded: ${dataset.length} rows, ${Object.keys(dataset[0]).length} columns`;

    // Create table
    const table = document.createElement('table');
    table.className = 'data-table';

    // Header
    const header = table.createTHead();
    const headerRow = header.insertRow();
    Object.keys(dataset[0]).forEach(col => {
        const th = document.createElement('th');
        th.textContent = col;
        headerRow.appendChild(th);
    });

    // Body (first 10 rows)
    const tbody = table.createTBody();
    dataset.slice(0, 10).forEach(row => {
        const tr = tbody.insertRow();
        Object.values(row).forEach(value => {
            const td = tr.insertCell();
            td.textContent = value !== null ? value : 'N/A';
        });
    });

    tableDiv.innerHTML = '';
    tableDiv.appendChild(table);
    previewDiv.classList.remove('hidden');
    previewDiv.classList.add('fade-in');
}

function showModelConfig() {
    const configDiv = document.getElementById('modelConfig');
    const targetSelect = document.getElementById('targetColumn');

    // Populate target column options
    targetSelect.innerHTML = '<option value="">Select target column...</option>';
    Object.keys(dataset[0]).forEach(col => {
        const option = document.createElement('option');
        option.value = col;
        option.textContent = col;
        targetSelect.appendChild(option);
    });

    configDiv.classList.remove('hidden');
    configDiv.classList.add('fade-in');
}

function updateTargetColumn() {
    targetColumn = document.getElementById('targetColumn').value;
    if (targetColumn) {
        featureColumns = Object.keys(dataset[0]).filter(col => col !== targetColumn);
    }
}

function updateModelType() {
    modelType = document.getElementById('modelType').value;
}

function updateTrainSize() {
    const trainSize = document.getElementById('trainSize').value;
    document.getElementById('trainSizeValue').textContent = trainSize + '%';
}

async function trainModel() {
    if (!targetColumn) {
        showStatus('Please select a target column.', 'error');
        return;
    }

    showTrainingSection();
    
    try {
        // Prepare data
        const { trainData, testData } = prepareData();
        
        // Create and train model
        model = await createModel(trainData.features, trainData.labels);
        
        // Evaluate model
        const evaluation = await evaluateModel(model, testData);
        
        // Calculate feature importance (SHAP-like)
        featureImportances = calculateFeatureImportance(model, trainData.features);
        
        showPerformanceMetrics(evaluation);
        showFeatureImportance();
        showPredictionSection();
        
    } catch (error) {
        showStatus('Error training model: ' + error.message, 'error');
    }
}

function prepareData() {
    // Clean data
    const cleanData = dataset.filter(row => 
        featureColumns.every(col => row[col] !== null && row[col] !== undefined) &&
        row[targetColumn] !== null && row[targetColumn] !== undefined
    );

    // Calculate statistics for normalization
    trainedStats = calculateStats(cleanData);

    // Normalize features
    const normalizedData = cleanData.map(row => {
        const normalizedRow = {};
        featureColumns.forEach(col => {
            const value = parseFloat(row[col]) || 0;
            normalizedRow[col] = (value - trainedStats[col].mean) / trainedStats[col].std;
        });
        normalizedRow[targetColumn] = row[targetColumn];
        return normalizedRow;
    });

    // Split data
    const trainSize = parseInt(document.getElementById('trainSize').value) / 100;
    const splitIndex = Math.floor(normalizedData.length * trainSize);
    
    // Shuffle data
    const shuffledData = normalizedData.sort(() => Math.random() - 0.5);
    
    const trainData = {
        features: shuffledData.slice(0, splitIndex).map(row => 
            featureColumns.map(col => row[col])
        ),
        labels: shuffledData.slice(0, splitIndex).map(row => 
            modelType === 'classification' ? (row[targetColumn] > 0 ? 1 : 0) : row[targetColumn]
        )
    };

    const testData = {
        features: shuffledData.slice(splitIndex).map(row => 
            featureColumns.map(col => row[col])
        ),
        labels: shuffledData.slice(splitIndex).map(row => 
            modelType === 'classification' ? (row[targetColumn] > 0 ? 1 : 0) : row[targetColumn]
        )
    };

    return { trainData, testData };
}

function calculateStats(data) {
    const stats = {};
    featureColumns.forEach(col => {
        const values = data.map(row => parseFloat(row[col]) || 0);
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
        const std = Math.sqrt(variance) || 1;
        stats[col] = { mean, std };
    });
    return stats;
}

async function createModel(features, labels) {
    const model = tf.sequential();
    
    if (modelType === 'classification') {
        model.add(tf.layers.dense({ inputShape: [featureColumns.length], units: 64, activation: 'relu' }));
        model.add(tf.layers.dropout({ rate: 0.2 }));
        model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
        model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
        
        model.compile({
            optimizer: 'adam',
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });
    } else {
        model.add(tf.layers.dense({ inputShape: [featureColumns.length], units: 64, activation: 'relu' }));
        model.add(tf.layers.dropout({ rate: 0.2 }));
        model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
        model.add(tf.layers.dense({ units: 1 }));
        
        model.compile({
            optimizer: 'adam',
            loss: 'meanSquaredError',
            metrics: ['mae']
        });
    }

    const xs = tf.tensor2d(features);
    const ys = tf.tensor2d(labels, [labels.length, 1]);

    await model.fit(xs, ys, {
        epochs: 50,
        validationSplit: 0.2,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                updateTrainingProgress((epoch + 1) / 50 * 100);
            }
        }
    });

    xs.dispose();
    ys.dispose();

    return model;
}

async function evaluateModel(model, testData) {
    const xs = tf.tensor2d(testData.features);
    const predictions = await model.predict(xs).data();
    
    let accuracy = 0;
    let mae = 0;
    let mse = 0;

    if (modelType === 'classification') {
        let correct = 0;
        for (let i = 0; i < predictions.length; i++) {
            const predicted = predictions[i] > 0.5 ? 1 : 0;
            if (predicted === testData.labels[i]) correct++;
        }
        accuracy = correct / predictions.length;
    } else {
        const errors = predictions.map((pred, i) => pred - testData.labels[i]);
        mae = errors.reduce((sum, err) => sum + Math.abs(err), 0) / errors.length;
        mse = errors.reduce((sum, err) => sum + err * err, 0) / errors.length;
    }

    xs.dispose();

    return { accuracy, mae, mse, predictions, actual: testData.labels };
}

function calculateFeatureImportance(model, features) {
    // Simplified feature importance calculation
    const importance = {};
    
    featureColumns.forEach((col, idx) => {
        // Calculate impact by perturbing feature values
        let totalImpact = 0;
        const sampleSize = Math.min(100, features.length);
        
        for (let i = 0; i < sampleSize; i++) {
            const original = [...features[i]];
            const perturbed = [...features[i]];
            perturbed[idx] = 0; // Zero out the feature
            
            const originalPred = model.predict(tf.tensor2d([original])).dataSync()[0];
            const perturbedPred = model.predict(tf.tensor2d([perturbed])).dataSync()[0];
            
            totalImpact += Math.abs(originalPred - perturbedPred);
        }
        
        importance[col] = totalImpact / sampleSize;
    });

    return importance;
}

function showTrainingSection() {
    const section = document.getElementById('trainingSection');
    section.classList.remove('hidden');
    section.classList.add('fade-in');
}

function updateTrainingProgress(progress) {
    const progressBar = document.getElementById('trainingProgress');
    progressBar.style.width = progress + '%';
    
    if (progress >= 100) {
        document.getElementById('trainingStatus').innerHTML = 
            '<div class="status success">✅ Model training completed!</div>';
    }
}

function showPerformanceMetrics(evaluation) {
    const section = document.getElementById('performanceSection');
    const metricsDiv = document.getElementById('performanceMetrics');
    
    let metricsHTML = '';
    
    if (modelType === 'classification') {
        metricsHTML = `
            <div class="metric-card">
                <div class="metric-value">${(evaluation.accuracy * 100).toFixed(1)}%</div>
                <div class="metric-label">Accuracy</div>
            </div>
        `;
    } else {
        metricsHTML = `
            <div class="metric-card">
                <div class="metric-value">${evaluation.mae.toFixed(3)}</div>
                <div class="metric-label">Mean Absolute Error</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${evaluation.mse.toFixed(3)}</div>
                <div class="metric-label">Mean Squared Error</div>
            </div>
        `;
    }
    
    metricsDiv.innerHTML = metricsHTML;
    section.classList.remove('hidden');
    section.classList.add('fade-in');
    
    // Create performance chart
    createPerformanceChart(evaluation);
}

function createPerformanceChart(evaluation) {
            const ctx = document.getElementById('performanceChart').getContext('2d');
            
            if (modelType === 'classification') {
                new Chart(ctx, {
                    type: 'scatter',
                    data: {
                        datasets: [{
                            label: 'Predictions vs Actual',
                            data: evaluation.predictions.map((pred, i) => ({
                                x: evaluation.actual[i],
                                y: pred
                            })),
                            backgroundColor: 'rgba(102, 126, 234, 0.6)',
                            borderColor: 'rgba(102, 126, 234, 1)',
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            x: { title: { display: true, text: 'Actual' } },
                            y: { title: { display: true, text: 'Predicted' } }
                        }
                    }
                });
            } else {
                new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: evaluation.actual.map((_, i) => i),
                        datasets: [{
                            label: 'Actual',
                            data: evaluation.actual,
                            borderColor: 'rgba(40, 167, 69, 1)',
                            backgroundColor: 'rgba(40, 167, 69, 0.1)',
                            fill: false
                        }, {
                            label: 'Predicted',
                            data: evaluation.predictions,
                            borderColor: 'rgba(102, 126, 234, 1)',
                            backgroundColor: 'rgba(102, 126, 234, 0.1)',
                            fill: false
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            x: { title: { display: true, text: 'Sample' } },
                            y: { title: { display: true, text: 'Value' } }
                        }
                    }
                });
            }
        }

        function showFeatureImportance() {
            const section = document.getElementById('importanceSection');
            const importanceDiv = document.getElementById('featureImportance');
            
            // Sort features by importance
            const sortedFeatures = Object.entries(featureImportances)
                .sort(([,a], [,b]) => b - a);
            
            const maxImportance = Math.max(...Object.values(featureImportances));
            
            let importanceHTML = '';
            sortedFeatures.forEach(([feature, importance]) => {
                const percentage = (importance / maxImportance) * 100;
                importanceHTML += `
                    <div class="feature-bar">
                        <div class="feature-name">${feature}</div>
                        <div class="importance-bar">
                            <div class="importance-fill" style="width: ${percentage}%"></div>
                        </div>
                        <div class="importance-value">${importance.toFixed(3)}</div>
                    </div>
                `;
            });
            
            importanceDiv.innerHTML = importanceHTML;
            section.classList.remove('hidden');
            section.classList.add('fade-in');
        }

        function showPredictionSection() {
            const section = document.getElementById('predictionSection');
            const inputsDiv = document.getElementById('predictionInputs');
            
            let inputsHTML = '';
            featureColumns.forEach(col => {
                inputsHTML += `
                    <div class="form-group">
                        <label for="input_${col}">${col}:</label>
                        <input type="number" id="input_${col}" placeholder="Enter ${col}" step="any">
                    </div>
                `;
            });
            
            inputsDiv.innerHTML = inputsHTML;
            section.classList.remove('hidden');
            section.classList.add('fade-in');
        }

        async function makePrediction() {
            // Get input values
            const inputValues = {};
            let hasError = false;
            
            featureColumns.forEach(col => {
                const input = document.getElementById(`input_${col}`);
                const value = parseFloat(input.value);
                if (isNaN(value)) {
                    hasError = true;
                    input.style.borderColor = '#dc3545';
                } else {
                    input.style.borderColor = '#ddd';
                    inputValues[col] = value;
                }
            });
            
            if (hasError) {
                showStatus('Please fill in all input fields with valid numbers.', 'error');
                return;
            }
            
            // Normalize inputs using training statistics
            const normalizedInputs = featureColumns.map(col => 
                (inputValues[col] - trainedStats[col].mean) / trainedStats[col].std
            );
            
            // Make prediction
            const prediction = await model.predict(tf.tensor2d([normalizedInputs])).data();
            const predictionValue = prediction[0];
            
            // Show prediction result
            showPredictionResult(predictionValue, inputValues);
            
            // Generate explanation
            generateExplanation(normalizedInputs, inputValues, predictionValue);
        }

        function showPredictionResult(prediction, inputs) {
            const resultDiv = document.getElementById('predictionResult');
            
            let resultHTML = '';
            if (modelType === 'classification') {
                const probability = (prediction * 100).toFixed(1);
                const prediction_class = prediction > 0.5 ? 'Positive' : 'Negative';
                resultHTML = `
                    <div class="prediction-result">
                        <h3>🎯 Prediction: ${prediction_class}</h3>
                        <p>Confidence: ${probability}%</p>
                    </div>
                `;
            } else {
                resultHTML = `
                    <div class="prediction-result">
                        <h3>🎯 Predicted Value: ${prediction.toFixed(3)}</h3>
                    </div>
                `;
            }
            
            resultDiv.innerHTML = resultHTML;
            resultDiv.classList.remove('hidden');
            resultDiv.classList.add('fade-in');
        }

        function generateExplanation(normalizedInputs, originalInputs, prediction) {
            const explanationDiv = document.getElementById('predictionExplanation');
            
            // Calculate feature contributions (simplified LIME-like explanation)
            const contributions = {};
            
            featureColumns.forEach((col, idx) => {
                // Calculate contribution by zeroing out the feature
                const zeroedInputs = [...normalizedInputs];
                zeroedInputs[idx] = 0;
                
                const originalPred = model.predict(tf.tensor2d([normalizedInputs])).dataSync()[0];
                const modifiedPred = model.predict(tf.tensor2d([zeroedInputs])).dataSync()[0];
                
                contributions[col] = {
                    impact: originalPred - modifiedPred,
                    value: originalInputs[col],
                    importance: featureImportances[col]
                };
            });
            
            // Sort contributions by absolute impact
            const sortedContributions = Object.entries(contributions)
                .sort(([,a], [,b]) => Math.abs(b.impact) - Math.abs(a.impact));
            
            let explanationHTML = `
                <h3>🔍 Prediction Explanation</h3>
                <p>Here's how each feature contributed to this prediction:</p>
                <div class="feature-importance">
            `;
            
            sortedContributions.forEach(([feature, data]) => {
                const impact = data.impact;
                const absImpact = Math.abs(impact);
                const maxAbsImpact = Math.max(...Object.values(contributions).map(c => Math.abs(c.impact)));
                const percentage = (absImpact / maxAbsImpact) * 100;
                const direction = impact > 0 ? 'increases' : 'decreases';
                const color = impact > 0 ? '#28a745' : '#dc3545';
                
                explanationHTML += `
                    <div class="feature-bar">
                        <div class="feature-name">${feature}</div>
                        <div class="importance-bar">
                            <div class="importance-fill" style="width: ${percentage}%; background: ${color}"></div>
                        </div>
                        <div class="importance-value">
                            ${impact > 0 ? '+' : ''}${impact.toFixed(3)}
                        </div>
                    </div>
                    <div style="margin-left: 10px; font-size: 12px; color: #666; margin-bottom: 10px;">
                        Value: ${data.value.toFixed(2)} → ${direction} prediction by ${absImpact.toFixed(3)}
                    </div>
                `;
            });
            
            explanationHTML += `
                </div>
                <div class="status info">
                    <h4>🧠 How to interpret this explanation:</h4>
                    <ul style="margin-left: 20px; margin-top: 10px;">
                        <li><strong>Green bars</strong>: Features that increase the prediction</li>
                        <li><strong>Red bars</strong>: Features that decrease the prediction</li>
                        <li><strong>Bar length</strong>: Shows the magnitude of each feature's impact</li>
                        <li><strong>Values</strong>: The actual input values you provided</li>
                    </ul>
                </div>
            `;
            
            explanationDiv.innerHTML = explanationHTML;
            explanationDiv.classList.remove('hidden');
            explanationDiv.classList.add('fade-in');
        }

        function showStatus(message, type) {
            // Create a temporary status message
            const statusDiv = document.createElement('div');
            statusDiv.className = `status ${type}`;
            statusDiv.textContent = message;
            
            // Insert at the top of the container
            const container = document.querySelector('.container');
            container.insertBefore(statusDiv, container.firstChild);
            
            // Remove after 5 seconds
            setTimeout(() => {
                if (statusDiv.parentNode) {
                    statusDiv.parentNode.removeChild(statusDiv);
                }
            }, 5000);
        }

        // Utility functions
        function downloadModel() {
            if (!model) {
                showStatus('No trained model available to download.', 'error');
                return;
            }
            
            model.save('downloads://explainable-ai-model');
            showStatus('Model downloaded successfully!', 'success');
        }

        function resetDashboard() {
            // Reset all variables
            dataset = null;
            model = null;
            featureColumns = [];
            targetColumn = '';
            trainedStats = null;
            featureImportances = null;
            
            // Hide all sections except upload
            document.querySelectorAll('.card:not(:first-child)').forEach(card => {
                card.classList.add('hidden');
            });
            
            // Reset file input
            document.getElementById('fileInput').value = '';
            
            // Clear data preview
            document.getElementById('dataPreview').classList.add('hidden');
            
            showStatus('Dashboard reset successfully!', 'success');
        }

        // Add some sample data generation for demo purposes
        function generateSampleData() {
            const sampleData = [];
            const features = ['age', 'income', 'education_years', 'experience'];
            
            for (let i = 0; i < 1000; i++) {
                const age = Math.floor(Math.random() * 40) + 25;
                const income = Math.floor(Math.random() * 80000) + 30000;
                const education = Math.floor(Math.random() * 10) + 12;
                const experience = Math.floor(Math.random() * 20) + 1;
                
                // Simple target: high earner if income > 60000 AND education > 16
                const target = (income > 60000 && education > 16) ? 1 : 0;
                
                sampleData.push({
                    age,
                    income,
                    education_years: education,
                    experience,
                    high_earner: target
                });
            }
            
            // Convert to CSV string
            const csvHeader = Object.keys(sampleData[0]).join(',');
            const csvRows = sampleData.map(row => Object.values(row).join(','));
            const csvString = csvHeader + '\n' + csvRows.join('\n');
            
            // Create blob and download
            const blob = new Blob([csvString], { type: 'text/csv' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'sample_data.csv';
            a.click();
            window.URL.revokeObjectURL(url);
            
            showStatus('Sample dataset generated and downloaded!', 'success');
        }

        // Add keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            if (e.ctrlKey || e.metaKey) {
                switch(e.key) {
                    case 'u':
                        e.preventDefault();
                        document.getElementById('fileInput').click();
                        break;
                    case 'r':
                        e.preventDefault();
                        resetDashboard();
                        break;
                    case 's':
                        e.preventDefault();
                        generateSampleData();
                        break;
                }
            }
        });

        // Add tooltips and help text
        function addTooltips() {
            const tooltips = {
                'targetColumn': 'The column you want to predict',
                'modelType': 'Classification for categories, Regression for continuous values',
                'trainSize': 'Percentage of data used for training (rest for testing)'
            };
            
            Object.entries(tooltips).forEach(([id, text]) => {
                const element = document.getElementById(id);
                if (element) {
                    element.title = text;
                }
            });
        }

        // Initialize tooltips when page loads
        setTimeout(addTooltips, 100);
    </script>
</body>
</html>
