import * as math from 'mathjs';

// Implémentation d'un vrai réseau de neurones
export class SimpleNeuralNetwork {
  inputSize: number;
  hiddenSize: number;
  outputSize: number;
  weights1: math.Matrix;
  bias1: math.Matrix;
  weights2: math.Matrix;
  bias2: math.Matrix;
  learningRate: number;
  examples: Array<{ input: number[]; target: number }>;

  constructor(inputSize: number, hiddenSize: number, outputSize: number) {
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
  
  // Fonction d'activation (sigmoid optimisée)
  sigmoid(x: math.Matrix): math.Matrix {
    // Implémentation optimisée de sigmoid pour éviter des calculs d'exponentielle coûteux
    return math.map(x, (val: number) => {
      // Limiter les valeurs extrêmes pour éviter les erreurs de calcul
      if (val < -10) return 0.00005;
      if (val > 10) return 0.99995;
      return 1 / (1 + Math.exp(-val));
    });
  }
  
  // Forward pass (prédiction optimisée)
  predict(input: number[] | number[][]): {
    prediction: number[] | number;
    hiddenActivations: number[] | number;
    outputActivations: number[] | number;
  } {
    try {
      // Vectoriser l'entrée (aplatir la grille 2D)
      const inputArray = Array.isArray(input[0]) ? (input as number[][]).flat() : (input as number[]);
      const inputVector = math.matrix([inputArray]);
      
      // Couche 1 (entrée -> cachée)
      const hidden = math.add(math.multiply(inputVector, this.weights1), this.bias1);
      const hiddenOutput = this.sigmoid(hidden);
      
      // Couche 2 (cachée -> sortie)
      const output = math.add(math.multiply(hiddenOutput, this.weights2), this.bias2);
      const prediction = this.sigmoid(output);
      
      // Extraire les résultats et les mettre en cache pour éviter des calculs répétés
      let predictionResult: number[] = [];
      let hiddenResult: number[] = [];
      
      try {
        const squeezedPrediction = math.squeeze(prediction);
        const squeezedHidden = math.squeeze(hiddenOutput);
        
        // Convertir les résultats en tableaux si nécessaire
        if ((squeezedPrediction as any)._data) {
          predictionResult = (squeezedPrediction as any)._data;
        } else if (typeof (squeezedPrediction as any).toArray === 'function') {
          predictionResult = (squeezedPrediction as any).toArray();
        } else if (Array.isArray(squeezedPrediction)) {
          predictionResult = squeezedPrediction as number[];
        } else {
          predictionResult = Array(this.outputSize).fill(0.25);
        }
        
        if ((squeezedHidden as any)._data) {
          hiddenResult = (squeezedHidden as any)._data;
        } else if (typeof (squeezedHidden as any).toArray === 'function') {
          hiddenResult = (squeezedHidden as any).toArray();
        } else if (Array.isArray(squeezedHidden)) {
          hiddenResult = squeezedHidden as number[];
        } else {
          hiddenResult = Array(this.hiddenSize).fill(0.5);
        }
      } catch (error) {
        console.error("Error processing results:", error);
        
        // Fallback si le traitement échoue
        predictionResult = Array(this.outputSize).fill(0.25);
        hiddenResult = Array(this.hiddenSize).fill(0.5);
      }
      
      return {
        prediction: predictionResult,
        hiddenActivations: hiddenResult,
        outputActivations: predictionResult // Même valeur que prediction pour économiser du calcul
      };
    } catch (error) {
      console.error("Error during prediction:", error);
      
      // Retourner des valeurs par défaut en cas d'erreur
      return {
        prediction: Array(this.outputSize).fill(0.25),
        hiddenActivations: Array(this.hiddenSize).fill(0.5),
        outputActivations: Array(this.outputSize).fill(0.25)
      };
    }
  }
  
  // Méthode privée pour effectuer l'entraînement sur un exemple sans l'ajouter
  private trainOnExample(flatInput: number[], target: number): number {
    try {
      // Vectoriser l'entrée
      const inputVector = math.matrix([flatInput]);
      
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
      
      // Backpropagation (couche de sortie -> couche cachée) - optimisé
      const sigmoidGradOutput = math.map(prediction, (p: number) => p * (1 - p)); // Dérivée de sigmoid
      const deltaOutput = math.dotMultiply(outputError, sigmoidGradOutput);
      
      // Mise à jour des poids (couche de sortie)
      const deltaWeights2 = math.multiply(math.transpose(hiddenOutput), deltaOutput);
      this.weights2 = math.add(this.weights2, math.multiply(deltaWeights2, this.learningRate));
      this.bias2 = math.add(this.bias2, math.multiply(deltaOutput, this.learningRate));
      
      // Backpropagation (couche cachée -> entrée) - optimisé
      const hiddenError = math.multiply(deltaOutput, math.transpose(this.weights2));
      const sigmoidGradHidden = math.map(hiddenOutput, (h: number) => h * (1 - h)); // Dérivée de sigmoid
      const deltaHidden = math.dotMultiply(hiddenError, sigmoidGradHidden);
      
      // Mise à jour des poids (couche cachée)
      const deltaWeights1 = math.multiply(math.transpose(inputVector), deltaHidden);
      this.weights1 = math.add(this.weights1, math.multiply(deltaWeights1, this.learningRate));
      this.bias1 = math.add(this.bias1, math.multiply(deltaHidden, this.learningRate));
      
      // Retourner l'erreur pour le suivi (calcul simplifié)
      return math.sum(math.abs(math.squeeze(outputError))) as number;
    } catch (error) {
      console.error("Error during example training:", error);
      return 0;
    }
  }

  // Entraînement avec un seul exemple (optimisé)
  train(input: number[] | number[][], target: number): number {
    // Valider la cible
    if (target < 0 || target >= this.outputSize) {
      console.error("Invalid target value:", target);
      return 0;
    }
    
    try {
      // Gérer intelligemment le stockage des exemples pour maintenir un équilibre
      const maxExamples = 100; // Augmenté à 100 pour améliorer la généralisation et la robustesse
      const maxPerDigit = 25; // 25 exemples par chiffre (4 chiffres x 25 = 100)
      
      if (this.examples.length > maxExamples) {
        // Compter combien d'exemples nous avons pour chaque chiffre
        const countByDigit = Array(this.outputSize).fill(0);
        this.examples.forEach(ex => {
          if (ex.target >= 0 && ex.target < this.outputSize) {
            countByDigit[ex.target]++;
          }
        });
        
        console.log("Distribution actuelle des exemples:", countByDigit);
        
        // Vérifier si le chiffre qu'on essaie d'ajouter est déjà sur-représenté
        if (countByDigit[target] >= maxPerDigit) {
          // Si ce chiffre est déjà sur-représenté, on remplace un exemple existant du même chiffre
          // Trouver tous les exemples de ce chiffre
          const examplesOfSameDigit = this.examples
            .map((ex, idx) => ({ ex, idx }))
            .filter(item => item.ex.target === target);
          
          // Si nous avons des exemples de ce chiffre, remplacer le plus ancien au lieu d'en ajouter un nouveau
          if (examplesOfSameDigit.length > 0) {
            // Trouver l'exemple le plus ancien de ce chiffre
            const oldestIdx = examplesOfSameDigit[0].idx;
            // Le remplacer par le nouvel exemple au lieu d'en ajouter un nouveau
            const flatInput = Array.isArray(input[0]) ? (input as number[][]).flat() : (input as number[]);
            this.examples[oldestIdx] = { input: [...flatInput], target };
            console.log(`Remplacé un exemple existant du chiffre ${target} au lieu d'en ajouter un nouveau`);
            
            // Continuer avec l'entraînement mais ne pas ajouter de nouvel exemple
            const inputVector = math.matrix([flatInput]);
            
            // Passer au calcul directement (skip l'ajout d'exemple)
            // Note: On retourne ici pour éviter d'ajouter un duplicata
            return this.trainOnExample(flatInput, target);
          }
        }
        
        // Si nous atteignons toujours la limite maximale, effectuer un élagage équilibré
        if (this.examples.length >= maxExamples) {
          // Pour chaque chiffre, garder les exemples les plus récents
          const newExamples = [];
          
          // Parcourir chaque chiffre
          for (let digit = 0; digit < this.outputSize; digit++) {
            // Obtenir tous les exemples de ce chiffre
            const examplesOfDigit = this.examples.filter(ex => ex.target === digit);
            
            // Garder les plus récents (jusqu'à maxPerDigit)
            const toKeep = examplesOfDigit.slice(-maxPerDigit);
            newExamples.push(...toKeep);
          }
          
          // Mettre à jour la liste d'exemples
          this.examples = newExamples;
          console.log("Exemples équilibrés. Nouvelle distribution:",
            Array(this.outputSize).fill(0).map((_, i) => 
              this.examples.filter(ex => ex.target === i).length
            )
          );
        }
      }
      
      // Préparer l'entrée aplatie
      const flatInput = Array.isArray(input[0]) ? (input as number[][]).flat() : (input as number[]);
      
      // Sauvegarder l'exemple
      this.examples.push({ input: [...flatInput], target });
      console.log(`Exemple ajouté - Total: ${this.examples.length}, Distribution:`,
        Array(this.outputSize).fill(0).map((_, i) => 
          this.examples.filter(ex => ex.target === i).length
        )
      );
      
      // Entraîner sur cet exemple en utilisant la méthode commune
      return this.trainOnExample(flatInput, target);
    } catch (error) {
      console.error("Error during training:", error);
      return 0;
    }
  }
  
  // Ré-entraîner sur tous les exemples précédents (optimisé et équilibré)
  retrainAll(epochs = 1): number {
    if (this.examples.length === 0) return 0;
    
    // Limiter le nombre d'exemples pour le réentraînement tout en gardant une bonne diversité
    const maxExamplesPerClass = 10; // Augmenté à 10 exemples par classe pour le réentraînement
    
    // Préparer un échantillon équilibré pour le réentraînement
    const samplesForRetraining: Array<{ input: number[]; target: number }> = [];
    
    // Pour chaque classe (chiffre), sélectionner des exemples de manière équilibrée
    for (let digit = 0; digit < this.outputSize; digit++) {
      // Trouver tous les exemples de cette classe
      const examplesOfClass = this.examples.filter(ex => ex.target === digit);
      
      if (examplesOfClass.length > 0) {
        // Si nous avons beaucoup d'exemples, en prendre un échantillon aléatoire
        let selectedExamples;
        if (examplesOfClass.length > maxExamplesPerClass) {
          // Mélanger et prendre un sous-ensemble
          selectedExamples = [...examplesOfClass]
            .sort(() => 0.5 - Math.random())
            .slice(0, maxExamplesPerClass);
        } else {
          // Sinon prendre tous les exemples disponibles
          selectedExamples = examplesOfClass;
        }
        
        // Ajouter les exemples sélectionnés à notre échantillon de réentraînement
        samplesForRetraining.push(...selectedExamples);
      }
    }
    
    // Afficher des informations sur l'équilibrage
    const distribution = Array(this.outputSize).fill(0);
    samplesForRetraining.forEach(ex => {
      if (ex.target >= 0 && ex.target < this.outputSize) {
        distribution[ex.target]++;
      }
    });
    console.log(`Réentraînement sur ${samplesForRetraining.length} exemples équilibrés. Distribution:`, distribution);
    
    // Effectuer le réentraînement sans ajouter les exemples à nouveau
    let totalError = 0;
    for (let e = 0; e < epochs; e++) {
      for (const example of samplesForRetraining) {
        // Utiliser trainOnExample au lieu de train pour éviter d'ajouter à nouveau l'exemple
        totalError += this.trainOnExample(example.input, example.target);
      }
    }
    
    return totalError / (epochs * samplesForRetraining.length || 1);
  }
  
  // Obtenir la prédiction la plus probable
  getPredictedLabel(input: number[] | number[][]): {
    label: number;
    confidence: number;
    allConfidences: number[];
    hiddenActivations: number[];
  } {
    const result = this.predict(input);
    let predArray: number[];
    
    try {
      if (result.prediction && (result.prediction as any)._data) {
        predArray = (result.prediction as any)._data;
      } else if (result.prediction && typeof (result.prediction as any).toArray === 'function') {
        predArray = (result.prediction as any).toArray();
      } else if (Array.isArray(result.prediction)) {
        predArray = result.prediction as number[];
      } else {
        predArray = Array(this.outputSize).fill(0);
      }
    } catch (error) {
      console.error("Error processing prediction result:", error);
      predArray = Array(this.outputSize).fill(0);
    }
    
    // Trouver l'indice avec la valeur maximale
    let maxIdx = 0;
    let maxVal = predArray[0] || 0;
    for (let i = 1; i < predArray.length; i++) {
      if (predArray[i] > maxVal) {
        maxVal = predArray[i];
        maxIdx = i;
      }
    }
    
    let hiddenActs: number[];
    try {
      if (result.hiddenActivations && (result.hiddenActivations as any)._data) {
        hiddenActs = (result.hiddenActivations as any)._data;
      } else if (result.hiddenActivations && typeof (result.hiddenActivations as any).toArray === 'function') {
        hiddenActs = (result.hiddenActivations as any).toArray();
      } else if (Array.isArray(result.hiddenActivations)) {
        hiddenActs = result.hiddenActivations as number[];
      } else {
        hiddenActs = Array(this.hiddenSize).fill(0.5);
      }
    } catch (error) {
      console.error("Error processing hidden activations:", error);
      hiddenActs = Array(this.hiddenSize).fill(0.5);
    }
    
    return {
      label: maxIdx,
      confidence: maxVal,
      allConfidences: predArray,
      hiddenActivations: hiddenActs
    };
  }
  
  // Sauvegarder le modèle (nouveau)
  serialize(): string {
    try {
      // Convertir les matrices mathjs en tableaux pour la sérialisation
      const matrixToArray = (matrix: any): number[][] => {
        if (!matrix) return [[]];
        try {
          if (typeof matrix.toArray === 'function') {
            return matrix.toArray();
          } else if (matrix._data) {
            return matrix._data;
          } else if (Array.isArray(matrix)) {
            return matrix;
          }
          return [[]];
        } catch (e) {
          console.error("Error converting matrix to array:", e);
          return [[]];
        }
      };
      
      return JSON.stringify({
        inputSize: this.inputSize,
        hiddenSize: this.hiddenSize,
        outputSize: this.outputSize,
        weights1: matrixToArray(this.weights1),
        bias1: matrixToArray(this.bias1),
        weights2: matrixToArray(this.weights2),
        bias2: matrixToArray(this.bias2),
        examples: this.examples
      });
    } catch (error) {
      console.error("Error serializing neural network:", error);
      throw new Error("Failed to serialize neural network");
    }
  }
  
  // Charger un modèle (nouveau)
  static deserialize(serialized: string): SimpleNeuralNetwork {
    try {
      const data = JSON.parse(serialized);
      
      // Vérification des propriétés requises
      if (!data.inputSize || !data.hiddenSize || !data.outputSize || 
          !data.weights1 || !data.bias1 || !data.weights2 || !data.bias2) {
        throw new Error("Invalid model format: missing required properties");
      }
      
      const network = new SimpleNeuralNetwork(data.inputSize, data.hiddenSize, data.outputSize);
      
      try {
        network.weights1 = math.matrix(data.weights1);
        network.bias1 = math.matrix(data.bias1);
        network.weights2 = math.matrix(data.weights2);
        network.bias2 = math.matrix(data.bias2);
        
        if (Array.isArray(data.examples)) {
          network.examples = data.examples;
        } else {
          network.examples = [];
        }
      } catch (error) {
        console.error("Error converting arrays to matrices:", error);
        throw new Error("Failed to convert model data to matrices");
      }
      
      return network;
    } catch (error) {
      console.error("Error deserializing neural network:", error);
      throw new Error("Failed to deserialize neural network");
    }
  }
}