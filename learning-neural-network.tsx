import React, { useState, useEffect } from 'react';
import { Play, RotateCcw, AlertTriangle, Check, X } from 'lucide-react';
import * as math from 'mathjs';

// Implémentation d'un vrai réseau de neurones
class SimpleNeuralNetwork {
  constructor(inputSize, hiddenSize, outputSize) {
    // Initialiser les poids avec les bonnes dimensions
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.outputSize = outputSize;
    
    // Initialiser avec des poids aléatoires
    this.weights1 = math.matrix(math.random([inputSize, hiddenSize], -0.5, 0.5));
    this.bias1 = math.matrix(math.zeros([1, hiddenSize]));
    this.weights2 = math.matrix(math.random([hiddenSize, outputSize], -0.5, 0.5));
    this.bias2 = math.matrix(math.zeros([1, outputSize]));
    
    this.learningRate = 0.1;
    this.examples = []; // Stockage des exemples d'entraînement
  }
  
  // Fonction d'activation (sigmoid)
  sigmoid(x) {
    return math.map(x, val => 1 / (1 + Math.exp(-val)));
  }
  
  // Forward pass (prédiction)
  predict(input) {
    // Vectoriser l'entrée (aplatir la grille 2D)
    const inputArray = Array.isArray(input[0]) ? input.flat() : input;
    const inputVector = math.matrix([inputArray]);
    
    // Couche 1 (entrée -> cachée)
    const hidden = math.add(math.multiply(inputVector, this.weights1), this.bias1);
    const hiddenOutput = this.sigmoid(hidden);
    
    // Couche 2 (cachée -> sortie)
    const output = math.add(math.multiply(hiddenOutput, this.weights2), this.bias2);
    const prediction = this.sigmoid(output);
    
    return {
      prediction: math.squeeze(prediction),
      hiddenActivations: math.squeeze(hiddenOutput),
      outputActivations: math.squeeze(prediction)
    };
  }
  
  // Entraînement avec un seul exemple
  train(input, target) {
    // Sauvegarder l'exemple
    this.examples.push({ input: [...input], target });
    
    // Vectoriser l'entrée
    const inputArray = Array.isArray(input[0]) ? input.flat() : input;
    const inputVector = math.matrix([inputArray]);
    
    // Créer le vecteur cible (one-hot encoding)
    const targetArray = Array(this.outputSize).fill(0);
    targetArray[target] = 1;
    const targetVector = math.matrix([targetArray]);
    
    // Forward pass
    const hidden = math.add(math.multiply(inputVector, this.weights1), this.bias1);
    const hiddenOutput = this.sigmoid(hidden);
    const output = math.add(math.multiply(hiddenOutput, this.weights2), this.bias2);
    const prediction = this.sigmoid(output);
    
    // Calcul de l'erreur (sortie)
    const outputError = math.subtract(targetVector, prediction);
    
    // Backpropagation (couche de sortie -> couche cachée)
    const sigmoidGradOutput = math.map(prediction, p => p * (1 - p)); // Dérivée de sigmoid
    const deltaOutput = math.dotMultiply(outputError, sigmoidGradOutput);
    
    // Backpropagation (couche cachée -> entrée)
    const hiddenError = math.multiply(deltaOutput, math.transpose(this.weights2));
    const sigmoidGradHidden = math.map(hiddenOutput, h => h * (1 - h)); // Dérivée de sigmoid
    const deltaHidden = math.dotMultiply(hiddenError, sigmoidGradHidden);
    
    // Mise à jour des poids (couche de sortie)
    const deltaWeights2 = math.multiply(math.transpose(hiddenOutput), deltaOutput);
    this.weights2 = math.add(this.weights2, math.multiply(deltaWeights2, this.learningRate));
    this.bias2 = math.add(this.bias2, math.multiply(deltaOutput, this.learningRate));
    
    // Mise à jour des poids (couche cachée)
    const deltaWeights1 = math.multiply(math.transpose(inputVector), deltaHidden);
    this.weights1 = math.add(this.weights1, math.multiply(deltaWeights1, this.learningRate));
    this.bias1 = math.add(this.bias1, math.multiply(deltaHidden, this.learningRate));
    
    // Retourner l'erreur pour le suivi
    return math.sum(math.abs(math.squeeze(outputError)));
  }
  
  // Ré-entraîner sur tous les exemples précédents
  retrainAll(epochs = 1) {
    if (this.examples.length === 0) return 0;
    
    let totalError = 0;
    for (let e = 0; e < epochs; e++) {
      for (const example of this.examples) {
        totalError += this.train(example.input, example.target);
      }
    }
    return totalError / (epochs * this.examples.length);
  }
  
  // Obtenir la prédiction la plus probable
  getPredictedLabel(input) {
    const result = this.predict(input);
    let predArray;
    
    if (result.prediction && result.prediction._data) {
      predArray = result.prediction._data;
    } else if (result.prediction && typeof result.prediction.toArray === 'function') {
      predArray = result.prediction.toArray();
    } else {
      predArray = [0, 0, 0, 0];
    }
    
    // Trouver l'indice avec la valeur maximale
    let maxIdx = 0;
    let maxVal = predArray[0];
    for (let i = 1; i < predArray.length; i++) {
      if (predArray[i] > maxVal) {
        maxVal = predArray[i];
        maxIdx = i;
      }
    }
    
    return {
      label: maxIdx,
      confidence: maxVal,
      allConfidences: predArray,
      hiddenActivations: result.hiddenActivations
    };
  }
}

// Composant principal
const LearningNeuralNetwork = () => {
  const gridSize = 8;
  const inputSize = gridSize * gridSize;
  const hiddenSize = 16;
  const outputSize = 4;
  
  // États pour l'interface
  const [grid, setGrid] = useState(Array(gridSize).fill().map(() => Array(gridSize).fill(0)));
  const [isDrawing, setIsDrawing] = useState(false);
  const [running, setRunning] = useState(false);
  const [network, setNetwork] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [hiddenActivations, setHiddenActivations] = useState(Array(hiddenSize).fill(0));
  const [outputActivations, setOutputActivations] = useState(Array(outputSize).fill(0));
  const [training, setTraining] = useState(false);
  const [examplesCount, setExamplesCount] = useState(0);
  const [phase, setPhase] = useState('idle'); // idle, recognizing, feedback, learning
  const [feedbackDigit, setFeedbackDigit] = useState(null);
  const [highConfidence, setHighConfidence] = useState(false);
  
  // Initialiser le réseau de neurones
  useEffect(() => {
    setNetwork(new SimpleNeuralNetwork(inputSize, hiddenSize, outputSize));
  }, []);
  
  // Fonctions pour gérer le dessin
  const startDrawing = () => {
    setIsDrawing(true);
  };
  
  const stopDrawing = () => {
    setIsDrawing(false);
  };
  
  const handleCellInteraction = (rowIndex, colIndex) => {
    if (!isDrawing) return;
    
    setGrid(prevGrid => {
      const newGrid = [...prevGrid.map(row => [...row])];
      // Basculer entre 0 (blanc) et 1 (noir)
      newGrid[rowIndex][colIndex] = newGrid[rowIndex][colIndex] ? 0 : 1;
      return newGrid;
    });
  };
  
  // Reconnaître le chiffre dessiné
  const recognizeDrawing = () => {
    if (!network || grid.flat().every(cell => cell === 0)) return;
    
    setPhase('recognizing');
    setRunning(true);
    
    // Simuler la progression de l'activation des neurones
    let step = 0;
    const interval = setInterval(() => {
      step++;
      if (step === 5) {
        clearInterval(interval);
        
        // Obtenir la prédiction réelle du réseau
        const flatGrid = grid.flat();
        const result = network.getPredictedLabel(flatGrid);
        
        // Mettre à jour les activations des neurones
        let hiddenActs;
        if (result.hiddenActivations && result.hiddenActivations._data) {
          hiddenActs = result.hiddenActivations._data;
        } else if (result.hiddenActivations && typeof result.hiddenActivations.toArray === 'function') {
          hiddenActs = result.hiddenActivations.toArray();
        } else {
          hiddenActs = Array(hiddenSize).fill(0.5);
        }
        setHiddenActivations(hiddenActs);
        setOutputActivations(result.allConfidences);
        
        // Déterminer si la confiance est suffisamment élevée
        const isConfident = result.confidence > 0.7;
        setHighConfidence(isConfident);
        
        // Définir la phase suivante en fonction de la confiance
        setPhase(isConfident ? 'feedback' : 'learning');
        setPrediction(result);
        setRunning(false);
      } else {
        // Animation progressive des neurones
        setHiddenActivations(prev => {
          const newAct = [...prev];
          for (let i = 0; i < hiddenSize / 2; i++) {
            const idx = Math.floor(Math.random() * hiddenSize);
            newAct[idx] = Math.min(1, newAct[idx] + Math.random() * 0.3);
          }
          return newAct;
        });
      }
    }, 200);
  };
  
  // Enseigner au réseau (quand il n'est pas confiant)
  const teachNetwork = (digit) => {
    if (!network) return;
    
    setTraining(true);
    setFeedbackDigit(digit);
    
    // Entraîner le réseau
    const flatGrid = grid.flat();
    network.train(flatGrid, digit);
    
    // Renforcer la mémoire
    network.retrainAll(2);
    
    // Mise à jour du compteur d'exemples
    setExamplesCount(network.examples.length);
    
    // Simuler le processus d'apprentissage
    setTimeout(() => {
      setTraining(false);
      setPhase('idle');
      
      // Mettre à jour la prédiction après l'apprentissage
      const newResult = network.getPredictedLabel(flatGrid);
      let hiddenActs;
      if (newResult.hiddenActivations && newResult.hiddenActivations._data) {
        hiddenActs = newResult.hiddenActivations._data;
      } else if (newResult.hiddenActivations && typeof newResult.hiddenActivations.toArray === 'function') {
        hiddenActs = newResult.hiddenActivations.toArray();
      } else {
        hiddenActs = Array(hiddenSize).fill(0.5);
      }
      setHiddenActivations(hiddenActs);
      setOutputActivations(newResult.allConfidences);
    }, 1000);
  };
  
  // Confirmer ou corriger la prédiction
  const provideFeedback = (isCorrect, correctDigit = null) => {
    if (!network || !prediction) return;
    
    if (isCorrect) {
      // Renforcer l'apprentissage même si c'était déjà correct
      teachNetwork(prediction.label);
    } else if (correctDigit !== null) {
      // Corriger avec le bon chiffre
      teachNetwork(correctDigit);
    }
  };
  
  // Réinitialiser le dessin
  const resetDrawing = () => {
    setGrid(Array(gridSize).fill().map(() => Array(gridSize).fill(0)));
    setHiddenActivations(Array(hiddenSize).fill(0));
    setOutputActivations(Array(outputSize).fill(0));
    setPrediction(null);
    setPhase('idle');
  };
  
  return (
    <div className="flex flex-col items-center p-6 bg-white rounded-lg shadow-lg w-full max-w-4xl mx-auto">
      <h2 className="text-2xl font-bold mb-2">Réseau de Neurones Interactif à Apprentissage Continu</h2>
      <p className="text-gray-600 mb-4">
        Dessinez un chiffre (0-3). Le réseau apprendra de vos corrections.
        {examplesCount > 0 && ` Exemples appris: ${examplesCount}`}
      </p>
      
      <div className="flex flex-wrap justify-center gap-8 w-full">
        {/* Colonne de gauche: grille de dessin */}
        <div className="flex flex-col items-center">
          <h3 className="text-lg font-semibold mb-2">Dessinez un chiffre (0-3)</h3>
          <div className="mb-4 border-2 border-gray-300 p-2 rounded-lg">
            <div 
              className="grid grid-cols-8 gap-0"
              onMouseDown={startDrawing}
              onMouseUp={stopDrawing}
              onMouseLeave={stopDrawing}
            >
              {grid.map((row, rowIndex) => (
                row.map((cell, colIndex) => (
                  <div
                    key={`${rowIndex}-${colIndex}`}
                    className={`w-8 h-8 border border-gray-200 ${
                      cell ? 'bg-black' : 'bg-white'
                    } cursor-pointer`}
                    onMouseDown={() => handleCellInteraction(rowIndex, colIndex)}
                    onMouseEnter={() => handleCellInteraction(rowIndex, colIndex)}
                  />
                ))
              ))}
            </div>
          </div>
          
          {/* Contrôles */}
          <div className="flex justify-center gap-4 mb-6">
            <button 
              onClick={recognizeDrawing}
              disabled={running || training || phase === 'feedback' || phase === 'learning'}
              className={`flex items-center gap-2 px-4 py-2 ${
                running || training || phase === 'feedback' || phase === 'learning'
                  ? 'bg-gray-300 cursor-not-allowed'
                  : 'bg-blue-500 text-white hover:bg-blue-600'
              } rounded`}
            >
              <Play size={16} />
              Reconnaître
            </button>
            <button 
              onClick={resetDrawing}
              className="flex items-center gap-2 px-4 py-2 bg-gray-200 rounded hover:bg-gray-300"
            >
              <RotateCcw size={16} />
              Effacer
            </button>
          </div>
          
          {/* Interface de feedback/apprentissage */}
          {phase === 'learning' && (
            <div className="mt-2 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
              <div className="flex items-center gap-2 text-yellow-700 mb-2">
                <AlertTriangle size={18} />
                <span className="font-semibold">Je ne suis pas sûr, aidez-moi à apprendre!</span>
              </div>
              <p className="text-sm mb-2">Quel chiffre avez-vous dessiné?</p>
              <div className="flex justify-center gap-2">
                {[0, 1, 2, 3].map(digit => (
                  <button
                    key={digit}
                    onClick={() => teachNetwork(digit)}
                    disabled={training}
                    className="w-8 h-8 flex items-center justify-center border rounded bg-white hover:bg-blue-100"
                  >
                    {digit}
                  </button>
                ))}
              </div>
            </div>
          )}
          
          {phase === 'feedback' && (
            <div className="mt-2 p-3 bg-blue-50 border border-blue-200 rounded-lg">
              <div className="flex items-center gap-2 text-blue-700 mb-2">
                <span className="font-semibold">
                  Je pense que c'est un{' '}
                  <span className="text-lg font-bold">{prediction?.label}</span>
                  {' '}({Math.round(prediction?.confidence * 100)}% de confiance)
                </span>
              </div>
              <p className="text-sm mb-2">Est-ce correct?</p>
              <div className="flex justify-center gap-3">
                <button
                  onClick={() => provideFeedback(true)}
                  disabled={training}
                  className="flex items-center gap-1 px-3 py-1 bg-green-100 hover:bg-green-200 text-green-800 rounded"
                >
                  <Check size={16} />
                  Oui
                </button>
                <button
                  onClick={() => setPhase('learning')}
                  disabled={training}
                  className="flex items-center gap-1 px-3 py-1 bg-red-100 hover:bg-red-200 text-red-800 rounded"
                >
                  <X size={16} />
                  Non
                </button>
              </div>
            </div>
          )}
          
          {training && (
            <div className="mt-2 p-3 bg-blue-50 border border-blue-200 rounded-lg">
              <div className="text-center text-blue-700">
                <span className="font-semibold">Apprentissage en cours...</span>
              </div>
            </div>
          )}
        </div>
        
        {/* Colonne de droite: visualisation du réseau */}
        <div className="flex flex-col items-center">
          <h3 className="text-lg font-semibold mb-2">Visualisation du Réseau</h3>
          
          <div className="relative h-96 w-64 border border-gray-200 rounded-lg p-4">
            {/* Couche d'entrée (juste une représentation) */}
            <div className="absolute top-4 left-4">
              <div className="text-xs text-gray-500 mb-1">Entrée (64 neurones)</div>
              <div className="w-12 h-24 border border-gray-300 rounded bg-gray-100 flex items-center justify-center">
                <div className="text-xs">8x8</div>
              </div>
            </div>
            
            {/* Couche cachée */}
            <div className="absolute top-12 left-1/2 transform -translate-x-1/2">
              <div className="text-xs text-gray-500 mb-1 text-center">Couche cachée</div>
              <div className="flex flex-col gap-1">
                {hiddenActivations.map((activation, idx) => (
                  <div
                    key={idx}
                    className="w-6 h-6 rounded-full border border-blue-400"
                    style={{
                      backgroundColor: `rgba(59, 130, 246, ${activation})`,
                      transition: 'background-color 0.3s'
                    }}
                  />
                ))}
              </div>
            </div>
            
            {/* Couche de sortie */}
            <div className="absolute bottom-4 right-4">
              <div className="text-xs text-gray-500 mb-1 text-center">Sortie</div>
              <div className="flex flex-col gap-4">
                {outputActivations.map((activation, idx) => (
                  <div key={idx} className="flex items-center gap-2">
                    <div
                      className="w-8 h-8 rounded-full border border-green-400 flex items-center justify-center font-bold"
                      style={{
                        backgroundColor: `rgba(52, 211, 153, ${activation})`,
                        transition: 'background-color 0.3s'
                      }}
                    >
                      {idx}
                    </div>
                    <div className="w-16 h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-green-500"
                        style={{ width: `${activation * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
            
            {/* Connexions (simplifiées) */}
            <svg className="absolute inset-0 w-full h-full" style={{ zIndex: -1 }}>
              {/* Connexions entre l'entrée et la couche cachée */}
              <line x1="28" y1="40" x2="100" y2="100" stroke="#CBD5E1" strokeWidth="1" />
              <line x1="28" y1="40" x2="100" y2="160" stroke="#CBD5E1" strokeWidth="1" />
              <line x1="28" y1="40" x2="100" y2="220" stroke="#CBD5E1" strokeWidth="1" />
              
              {/* Connexions entre la couche cachée et la sortie */}
              <line x1="120" y1="100" x2="212" y2="300" stroke="#CBD5E1" strokeWidth="1" />
              <line x1="120" y1="160" x2="212" y2="340" stroke="#CBD5E1" strokeWidth="1" />
              <line x1="120" y1="220" x2="212" y2="260" stroke="#CBD5E1" strokeWidth="1" />
            </svg>
          </div>
          
          {/* Informations sur le modèle */}
          <div className="mt-4 text-sm text-gray-600">
            <p>Structure du réseau :</p>
            <ul className="list-disc pl-5 mt-1">
              <li>64 neurones d'entrée (grille 8×8)</li>
              <li>16 neurones cachés</li>
              <li>4 neurones de sortie (0-3)</li>
            </ul>
          </div>
        </div>
      </div>
      
      <div className="mt-6 text-sm text-gray-600 max-w-lg text-center">
        <p>Ce système utilise un vrai réseau de neurones qui apprend de vos interactions.</p>
        <p className="mt-1">Plus vous lui enseignez de chiffres, plus il deviendra précis!</p>
      </div>
    </div>
  );
};

export default LearningNeuralNetwork;