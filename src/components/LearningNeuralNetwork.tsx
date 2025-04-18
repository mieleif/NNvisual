import React, { useState, useEffect, useCallback } from 'react';
import { Play, RotateCcw, AlertTriangle, Check, X, Save, Upload, BarChart2 } from 'lucide-react';
import { SimpleNeuralNetwork } from '@/utils/SimpleNeuralNetwork';
import ModelVisualization from './ModelVisualization';

type GridType = number[][];
type Phase = 'idle' | 'recognizing' | 'feedback' | 'learning';

const LearningNeuralNetwork = () => {
  const gridSize = 8;
  const inputSize = gridSize * gridSize;
  const hiddenSize = 16;
  const outputSize = 4;
  
  // États pour l'interface
  const [grid, setGrid] = useState<GridType>(() => 
    Array(gridSize).fill(0).map(() => Array(gridSize).fill(0))
  );
  const [isDrawing, setIsDrawing] = useState(false);
  const [running, setRunning] = useState(false);
  const [network, setNetwork] = useState<SimpleNeuralNetwork | null>(null);
  const [prediction, setPrediction] = useState<{
    label: number;
    confidence: number;
    allConfidences: number[];
    hiddenActivations: number[];
  } | null>(null);
  const [hiddenActivations, setHiddenActivations] = useState<number[]>(Array(hiddenSize).fill(0));
  const [outputActivations, setOutputActivations] = useState<number[]>(Array(outputSize).fill(0));
  const [training, setTraining] = useState(false);
  const [trainingStep, setTrainingStep] = useState(0); // Ajout d'un état pour suivre l'étape d'apprentissage
  const [examplesCount, setExamplesCount] = useState(0);
  const [phase, setPhase] = useState<Phase>('idle');
  const [feedbackDigit, setFeedbackDigit] = useState<number | null>(null);
  const [highConfidence, setHighConfidence] = useState(false);
  const [showModelVisualization, setShowModelVisualization] = useState(false);
  
  // Initialiser le réseau de neurones
  useEffect(() => {
    // Essayer de charger le modèle enregistré localement
    const savedModel = localStorage.getItem('neural-network-model');
    
    if (savedModel) {
      try {
        const loadedNetwork = SimpleNeuralNetwork.deserialize(savedModel);
        setNetwork(loadedNetwork);
        // Met à jour le compteur d'exemples immédiatement après le chargement
        setExamplesCount(loadedNetwork.examples.length);
        console.log('Modèle chargé avec succès:', loadedNetwork.examples.length, 'exemples');
      } catch (error) {
        console.error('Erreur lors du chargement du modèle:', error);
        const newNetwork = new SimpleNeuralNetwork(inputSize, hiddenSize, outputSize);
        setNetwork(newNetwork);
        setExamplesCount(0);
      }
    } else {
      const newNetwork = new SimpleNeuralNetwork(inputSize, hiddenSize, outputSize);
      setNetwork(newNetwork);
      setExamplesCount(0);
    }
  }, [inputSize, hiddenSize, outputSize]);
  
  // Effet pour mettre à jour le compteur d'exemples quand le réseau change
  useEffect(() => {
    if (network) {
      setExamplesCount(network.examples.length);
    }
  }, [network]);
  
  // Fonctions pour gérer le dessin
  const startDrawing = useCallback(() => {
    setIsDrawing(true);
  }, []);
  
  const stopDrawing = useCallback(() => {
    setIsDrawing(false);
  }, []);
  
  const handleCellInteraction = useCallback((rowIndex: number, colIndex: number) => {
    if (!isDrawing) return;
    
    setGrid(prevGrid => {
      const newGrid = [...prevGrid.map(row => [...row])];
      // Basculer entre 0 (blanc) et 1 (noir)
      newGrid[rowIndex][colIndex] = newGrid[rowIndex][colIndex] ? 0 : 1;
      return newGrid;
    });
  }, [isDrawing]);
  
  // Sauvegarder le modèle
  const saveModel = useCallback(() => {
    if (!network) return;
    
    try {
      const serialized = network.serialize();
      localStorage.setItem('neural-network-model', serialized);
      alert('Modèle sauvegardé avec succès!');
    } catch (error) {
      console.error('Erreur lors de la sauvegarde du modèle:', error);
      alert('Erreur lors de la sauvegarde du modèle.');
    }
  }, [network]);
  
  // Importer un modèle
  const importModel = useCallback(() => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    
    input.onchange = (event) => {
      const file = (event.target as HTMLInputElement).files?.[0];
      if (!file) return;
      
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const contents = e.target?.result as string;
          const loadedNetwork = SimpleNeuralNetwork.deserialize(contents);
          setNetwork(loadedNetwork);
          // Mise à jour fiable du compteur d'exemples
          if (loadedNetwork && loadedNetwork.examples) {
            setExamplesCount(loadedNetwork.examples.length);
            console.log('Exemples chargés:', loadedNetwork.examples.length);
          }
          localStorage.setItem('neural-network-model', contents);
          alert('Modèle importé avec succès!');
        } catch (error) {
          console.error('Erreur lors de l\'importation du modèle:', error);
          alert('Format de fichier invalide.');
        }
      };
      reader.readAsText(file);
    };
    
    input.click();
  }, []);
  
  // Exporter le modèle
  const exportModel = useCallback(() => {
    if (!network) return;
    
    try {
      const serialized = network.serialize();
      const blob = new Blob([serialized], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      
      const a = document.createElement('a');
      a.href = url;
      a.download = 'neural-network-model.json';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Erreur lors de l\'exportation du modèle:', error);
      alert('Erreur lors de l\'exportation du modèle.');
    }
  }, [network]);
  
  // Reconnaître le chiffre dessiné
  const recognizeDrawing = useCallback(() => {
    if (!network || grid.flat().every(cell => cell === 0)) return;
    
    setPhase('recognizing');
    setRunning(true);
    
    // Simuler la progression de l'activation des neurones
    let step = 0;
    const animationSteps = 5;
    const interval = setInterval(() => {
      step++;
      if (step === animationSteps) {
        clearInterval(interval);
        
        try {
          // Obtenir la prédiction réelle du réseau
          const flatGrid = grid.flat();
          const result = network.getPredictedLabel(flatGrid);
          
          // Mettre à jour les activations des neurones
          setHiddenActivations(result.hiddenActivations || Array(hiddenSize).fill(0));
          setOutputActivations(result.allConfidences || Array(outputSize).fill(0));
          
          // Déterminer si la confiance est suffisamment élevée
          const isConfident = result.confidence > 0.7;
          setHighConfidence(isConfident);
          
          // Définir la phase suivante en fonction de la confiance
          setPhase(isConfident ? 'feedback' : 'learning');
          setPrediction(result);
        } catch (error) {
          console.error("Error during neural network prediction:", error);
          setPhase('idle');
          setPrediction(null);
        } finally {
          setRunning(false);
        }
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
    
    return () => clearInterval(interval);
  }, [network, grid, hiddenSize]);
  
  // Enseigner au réseau (quand il n'est pas confiant)
  const teachNetwork = useCallback((digit: number) => {
    if (!network) return;
    
    // Valider la valeur du chiffre
    if (digit < 0 || digit >= outputSize) {
      console.error("Invalid digit value:", digit);
      return;
    }
    
    setTraining(true);
    setFeedbackDigit(digit);
    
    // Utilisation d'un Worker simulé via setTimeout pour ne pas bloquer l'UI
    // Étape 0: préparation
    setTrainingStep(0);
    setTimeout(() => {
      try {
        // Étape 1: préparation des calculs
        setTrainingStep(1);
        // Animation des neurones pendant l'entraînement
        setHiddenActivations(prev => prev.map(() => Math.random() * 0.5));
        
        // Étape 2: entraînement initial
        setTimeout(() => {
          try {
            setTrainingStep(2);
            const flatGrid = grid.flat();
            // Exécution rapide pour éviter de bloquer trop longtemps
            requestAnimationFrame(() => {
              try {
                network.train(flatGrid, digit);
                
                // Animation des neurones après apprentissage initial
                setHiddenActivations(prev => prev.map(() => Math.random() * 0.7));
                
                // Étape 3: renforcement de la mémoire
                setTimeout(() => {
                  try {
                    setTrainingStep(3);
                    // Exécution rapide pour éviter de bloquer trop longtemps
                    requestAnimationFrame(() => {
                      try {
                        // Renforcer la mémoire
                        network.retrainAll(1); // Réduit à 1 pour améliorer les performances
                        
                        // Mise à jour du compteur d'exemples de manière fiable
                        if (network && network.examples) {
                          // Force une mise à jour du compteur pour s'assurer qu'il est correct
                          setExamplesCount(network.examples.length);
                        }
                        
                        // Animation de fin d'apprentissage
                        setHiddenActivations(prev => prev.map(() => Math.random() * 0.9));
                        
                        // Étape 4: finalisation
                        setTimeout(() => {
                          try {
                            setTrainingStep(4);
                            // Exécution rapide pour éviter de bloquer trop longtemps
                            requestAnimationFrame(() => {
                              try {
                                // Mettre à jour la prédiction après l'apprentissage
                                const newResult = network.getPredictedLabel(flatGrid);
                                setHiddenActivations(newResult.hiddenActivations || Array(hiddenSize).fill(0));
                                setOutputActivations(newResult.allConfidences || Array(outputSize).fill(0));
                              } catch (error) {
                                console.error("Error updating predictions after training:", error);
                              } finally {
                                setTrainingStep(5); // Terminé
                                setTimeout(() => {
                                  setTraining(false);
                                  setPhase('idle');
                                  setTrainingStep(0);
                                }, 300);
                              }
                            });
                          } catch (error) {
                            console.error("Error in finalization step:", error);
                            setTraining(false);
                            setPhase('idle');
                            setTrainingStep(0);
                          }
                        }, 200);
                      } catch (error) {
                        console.error("Error during memory reinforcement:", error);
                        setTraining(false);
                        setPhase('idle');
                        setTrainingStep(0);
                      }
                    });
                  } catch (error) {
                    console.error("Error preparing memory reinforcement:", error);
                    setTraining(false);
                    setPhase('idle');
                    setTrainingStep(0);
                  }
                }, 200);
              } catch (error) {
                console.error("Error during neural network training:", error);
                setTraining(false);
                setPhase('idle');
                setTrainingStep(0);
              }
            });
          } catch (error) {
            console.error("Error preparing neural network training:", error);
            setTraining(false);
            setPhase('idle');
            setTrainingStep(0);
          }
        }, 200);
      } catch (error) {
        console.error("Error preparing training:", error);
        setTraining(false);
        setPhase('idle');
        setTrainingStep(0);
      }
    }, 10);
    
    // Pas de valeur de retour de nettoyage car les timeouts se terminent naturellement
  }, [network, grid, hiddenSize, outputSize]);
  
  // Confirmer ou corriger la prédiction
  const provideFeedback = useCallback((isCorrect: boolean, correctDigit: number | null = null) => {
    if (!network || !prediction) {
      console.warn("Cannot provide feedback: network or prediction is null");
      return;
    }
    
    try {
      if (isCorrect) {
        // Renforcer l'apprentissage même si c'était déjà correct
        teachNetwork(prediction.label);
      } else if (correctDigit !== null) {
        // Corriger avec le bon chiffre
        teachNetwork(correctDigit);
      } else {
        // Si pas correct mais pas de correction fournie, passer en mode apprentissage
        setPhase('learning');
      }
    } catch (error) {
      console.error("Error providing feedback:", error);
      setPhase('idle');
    }
  }, [network, prediction, teachNetwork]);
  
  // Réinitialiser le dessin
  const resetDrawing = useCallback(() => {
    setGrid(Array(gridSize).fill(0).map(() => Array(gridSize).fill(0)));
    setHiddenActivations(Array(hiddenSize).fill(0));
    setOutputActivations(Array(outputSize).fill(0));
    setPrediction(null);
    setPhase('idle');
  }, [gridSize, hiddenSize, outputSize]);
  
  return (
    <div className="flex flex-col items-center p-6 bg-white rounded-lg shadow-lg w-full max-w-4xl mx-auto">
      <h2 className="text-2xl font-bold mb-2">Réseau de Neurones Interactif à Apprentissage Continu</h2>
      <p className="text-gray-600 mb-4">
        Dessinez un chiffre (0-3). Le réseau apprendra de vos corrections.
        {examplesCount > 0 && ` Exemples appris: ${examplesCount}/100`}
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
              onTouchStart={startDrawing}
              onTouchEnd={stopDrawing}
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
                    onTouchStart={() => handleCellInteraction(rowIndex, colIndex)}
                    onTouchMove={() => handleCellInteraction(rowIndex, colIndex)}
                  />
                ))
              ))}
            </div>
          </div>
          
          {/* Contrôles */}
          <div className="flex flex-wrap justify-center gap-2 mb-6">
            <button 
              onClick={recognizeDrawing}
              disabled={running || training || phase === 'feedback' || phase === 'learning'}
              className={`flex items-center gap-2 px-3 py-2 ${
                running || training || phase === 'feedback' || phase === 'learning'
                  ? 'bg-gray-300 cursor-not-allowed'
                  : 'bg-blue-500 text-white hover:bg-blue-600'
              } rounded text-sm`}
            >
              <Play size={16} />
              Reconnaître
            </button>
            <button 
              onClick={resetDrawing}
              className="flex items-center gap-2 px-3 py-2 bg-gray-200 rounded hover:bg-gray-300 text-sm"
            >
              <RotateCcw size={16} />
              Effacer
            </button>
            <button 
              onClick={saveModel}
              className="flex items-center gap-2 px-3 py-2 bg-green-100 text-green-800 rounded hover:bg-green-200 text-sm"
              title="Sauvegarder le modèle dans le stockage local"
            >
              <Save size={16} />
              Sauvegarder
            </button>
            <div className="flex">
              <button 
                onClick={importModel}
                className="flex items-center gap-2 px-3 py-2 bg-blue-100 text-blue-800 rounded-l hover:bg-blue-200 text-sm"
                title="Importer un modèle depuis un fichier"
              >
                <Upload size={16} />
                Importer
              </button>
              <button 
                onClick={exportModel}
                className="flex items-center gap-2 px-3 py-2 bg-blue-100 text-blue-800 rounded-r hover:bg-blue-200 text-sm border-l border-blue-200"
                title="Exporter le modèle vers un fichier"
              >
                Exporter
              </button>
            </div>
            <button 
              onClick={() => setShowModelVisualization(true)}
              disabled={!network}
              className={`flex items-center gap-2 px-3 py-2 
                ${!network ? 'bg-gray-200 cursor-not-allowed' : 'bg-purple-100 text-purple-800 hover:bg-purple-200'} 
                rounded text-sm`}
              title="Visualiser les poids et les statistiques du modèle"
            >
              <BarChart2 size={16} />
              Visualiser
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
                    className={`w-8 h-8 flex items-center justify-center border rounded 
                      ${training ? 'bg-gray-100 cursor-not-allowed' : 'bg-white hover:bg-blue-100'}`}
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
                  {' '}({Math.round((prediction?.confidence || 0) * 100)}% de confiance)
                </span>
              </div>
              <p className="text-sm mb-2">Est-ce correct?</p>
              <div className="flex justify-center gap-3">
                <button
                  onClick={() => provideFeedback(true)}
                  disabled={training}
                  className={`flex items-center gap-1 px-3 py-1 
                    ${training ? 'bg-gray-100 cursor-not-allowed' : 'bg-green-100 hover:bg-green-200'} 
                    text-green-800 rounded`}
                >
                  <Check size={16} />
                  Oui
                </button>
                <button
                  onClick={() => provideFeedback(false)}
                  disabled={training}
                  className={`flex items-center gap-1 px-3 py-1 
                    ${training ? 'bg-gray-100 cursor-not-allowed' : 'bg-red-100 hover:bg-red-200'} 
                    text-red-800 rounded`}
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
              <div className="mt-2">
                <div className="h-2 w-full bg-gray-200 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-blue-500 transition-all duration-300" 
                    style={{ width: `${Math.min(100, trainingStep * 20)}%` }}
                  ></div>
                </div>
                <div className="flex justify-between mt-1 text-xs text-gray-500">
                  <span>
                    {trainingStep === 0 && "Préparation..."}
                    {trainingStep === 1 && "Analyse de l'exemple..."}
                    {trainingStep === 2 && "Entraînement..."}
                    {trainingStep === 3 && "Renforcement de la mémoire..."}
                    {trainingStep === 4 && "Finalisation..."}
                    {trainingStep === 5 && "Terminé!"}
                  </span>
                  <span>{Math.min(100, trainingStep * 20)}%</span>
                </div>
              </div>
            </div>
          )}
        </div>
        
        {/* Colonne de droite: visualisation du réseau */}
        <div className="flex flex-col items-center">
          <h3 className="text-lg font-semibold mb-2">Visualisation du Réseau</h3>
          
          <div className="relative h-64 w-full border border-gray-200 rounded-lg p-4">
            <div className="flex justify-between items-center h-full">
              {/* Couche d'entrée (gauche) */}
              <div className="flex flex-col items-center">
                <div className="text-xs text-gray-500 mb-2">Entrée (64 neurones)</div>
                <div className="w-16 h-32 border border-gray-300 rounded bg-gray-100 flex items-center justify-center">
                  <div className="text-xs">8x8</div>
                </div>
              </div>
              
              {/* Connexions entrée -> cachée */}
              <svg className="h-full w-24" style={{ position: 'relative', top: '8px' }}>
                <line x1="0" y1="80" x2="100%" y2="40" stroke="#CBD5E1" strokeWidth="1" />
                <line x1="0" y1="80" x2="100%" y2="80" stroke="#CBD5E1" strokeWidth="1" />
                <line x1="0" y1="80" x2="100%" y2="120" stroke="#CBD5E1" strokeWidth="1" />
              </svg>
              
              {/* Couche cachée (milieu) */}
              <div className="flex flex-col items-center">
                <div className="text-xs text-gray-500 mb-2 text-center">Couche cachée</div>
                <div className="grid grid-cols-4 gap-1" style={{ width: '100px' }}>
                  {hiddenActivations.map((activation, idx) => (
                    <div
                      key={idx}
                      className="w-6 h-6 rounded-full border border-blue-400"
                      style={{
                        backgroundColor: `rgba(59, 130, 246, ${activation})`,
                        transition: 'background-color 0.3s'
                      }}
                      title={`Neurone caché ${idx+1}: ${(activation * 100).toFixed(0)}% activé`}
                    />
                  ))}
                </div>
              </div>
              
              {/* Connexions cachée -> sortie */}
              <svg className="h-full w-24" style={{ position: 'relative', top: '8px' }}>
                <line x1="0" y1="40" x2="100%" y2="40" stroke="#CBD5E1" strokeWidth="1" />
                <line x1="0" y1="80" x2="100%" y2="80" stroke="#CBD5E1" strokeWidth="1" />
                <line x1="0" y1="120" x2="100%" y2="120" stroke="#CBD5E1" strokeWidth="1" />
              </svg>
              
              {/* Couche de sortie (droite) */}
              <div className="flex flex-col items-center">
                <div className="text-xs text-gray-500 mb-2 text-center">Sortie (0-3)</div>
                <div className="flex flex-col gap-3" style={{ marginTop: '10px' }}>
                  {outputActivations.map((activation, idx) => (
                    <div key={idx} className="flex items-center gap-2">
                      <div
                        className="w-8 h-8 rounded-full border border-green-400 flex items-center justify-center font-bold"
                        style={{
                          backgroundColor: `rgba(52, 211, 153, ${activation})`,
                          transition: 'background-color 0.3s'
                        }}
                        title={`Neurone de sortie ${idx}: ${(activation * 100).toFixed(1)}% activé`}
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
            </div>
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
      
      {/* Modale de visualisation du modèle */}
      {showModelVisualization && (
        <ModelVisualization 
          network={network} 
          onClose={() => setShowModelVisualization(false)} 
        />
      )}
    </div>
  );
};

export default LearningNeuralNetwork;