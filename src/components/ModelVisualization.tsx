import React from 'react';
import { SimpleNeuralNetwork } from '@/utils/SimpleNeuralNetwork';
import { X } from 'lucide-react';

interface ModelVisualizationProps {
  network: SimpleNeuralNetwork | null;
  onClose: () => void;
}

const ModelVisualization: React.FC<ModelVisualizationProps> = ({ network, onClose }) => {
  if (!network) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white p-6 rounded-lg shadow-lg max-w-4xl w-full max-h-[90vh] overflow-auto">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-bold">Visualisation du Modèle</h2>
            <button onClick={onClose} className="p-1 hover:bg-gray-200 rounded-full">
              <X size={24} />
            </button>
          </div>
          <p className="text-gray-600">Aucun modèle n'est disponible pour la visualisation.</p>
        </div>
      </div>
    );
  }

  // Extraire les poids et matrices pour la visualisation
  const weights1Array = getMatrixArray(network.weights1);
  const weights2Array = getMatrixArray(network.weights2);
  const bias1Array = getMatrixArray(network.bias1);
  const bias2Array = getMatrixArray(network.bias2);

  // Calculer les statistiques des poids
  const weights1Stats = calculateStats(weights1Array.flat());
  const weights2Stats = calculateStats(weights2Array.flat());

  // Extraire les exemples d'apprentissage
  const examples = network.examples;
  
  // Obtenir les distributions d'exemples par chiffre
  const exampleDistribution = [0, 1, 2, 3].map(digit => 
    examples.filter(ex => ex.target === digit).length
  );

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white p-6 rounded-lg shadow-lg max-w-4xl w-full max-h-[90vh] overflow-auto">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold">Visualisation du Modèle</h2>
          <button onClick={onClose} className="p-1 hover:bg-gray-200 rounded-full">
            <X size={24} />
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Statistiques du modèle */}
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="text-lg font-semibold mb-2">Statistiques du Modèle</h3>
            <ul className="space-y-2">
              <li><span className="font-medium">Entrée:</span> {network.inputSize} neurones</li>
              <li><span className="font-medium">Couche cachée:</span> {network.hiddenSize} neurones</li>
              <li><span className="font-medium">Sortie:</span> {network.outputSize} neurones</li>
              <li><span className="font-medium">Exemples appris:</span> {examples.length}</li>
              <li><span className="font-medium">Taux d'apprentissage:</span> {network.learningRate}</li>
            </ul>
          </div>

          {/* Distribution des exemples */}
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="text-lg font-semibold mb-2">Distribution des Exemples</h3>
            <div className="space-y-2">
              {[0, 1, 2, 3].map(digit => (
                <div key={digit} className="flex items-center">
                  <span className="w-8 text-center font-medium">{digit}:</span>
                  <div className="flex-1 h-5 bg-gray-200 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-blue-500"
                      style={{ 
                        width: examples.length > 0 
                          ? `${(exampleDistribution[digit] / examples.length) * 100}%`
                          : '0%'
                      }}
                    ></div>
                  </div>
                  <span className="ml-2 text-gray-600 text-sm">{exampleDistribution[digit]}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Visualisation des poids (entrée -> cachée) */}
          <div className="bg-gray-50 p-4 rounded-lg col-span-1 md:col-span-2">
            <h3 className="text-lg font-semibold mb-2">Poids: Entrée → Couche Cachée</h3>
            <p className="text-sm text-gray-600 mb-2">
              Min: {weights1Stats.min.toFixed(3)}, 
              Max: {weights1Stats.max.toFixed(3)}, 
              Moyenne: {weights1Stats.mean.toFixed(3)}
            </p>
            <div className="overflow-x-auto">
              <div className="weights-grid" style={{ 
                display: 'grid',
                gridTemplateColumns: `repeat(${network.hiddenSize}, minmax(20px, 1fr))`,
                gap: '1px'
              }}>
                {weights1Array.map((row, rowIdx) => 
                  row.map((weight, colIdx) => (
                    <div 
                      key={`${rowIdx}-${colIdx}`}
                      style={{
                        width: '20px',
                        height: '20px',
                        backgroundColor: getWeightColor(weight, weights1Stats.min, weights1Stats.max),
                      }}
                      title={`Entrée ${rowIdx} → Caché ${colIdx}: ${weight.toFixed(3)}`}
                    />
                  ))
                )}
              </div>
            </div>
            <p className="text-xs text-gray-500 mt-1">
              Chaque carré représente un poids entre un neurone d'entrée (ligne) et un neurone caché (colonne).
              Noir = poids négatif fort, Blanc = poids positif fort, Gris = proche de zéro.
            </p>
          </div>

          {/* Visualisation des poids (cachée -> sortie) */}
          <div className="bg-gray-50 p-4 rounded-lg col-span-1 md:col-span-2">
            <h3 className="text-lg font-semibold mb-2">Poids: Couche Cachée → Sortie</h3>
            <p className="text-sm text-gray-600 mb-2">
              Min: {weights2Stats.min.toFixed(3)}, 
              Max: {weights2Stats.max.toFixed(3)}, 
              Moyenne: {weights2Stats.mean.toFixed(3)}
            </p>
            <div className="flex justify-center">
              <div className="grid" style={{
                display: 'grid',
                gridTemplateColumns: `repeat(${network.outputSize}, minmax(60px, 1fr))`,
                gap: '8px'
              }}>
                {[...Array(network.outputSize)].map((_, outputIdx) => (
                  <div key={outputIdx} className="text-center">
                    <div className="font-medium mb-1">Sortie {outputIdx}</div>
                    <div className="grid grid-cols-4 gap-1">
                      {weights2Array.map((row, hiddenIdx) => (
                        <div 
                          key={hiddenIdx}
                          style={{
                            width: '20px',
                            height: '20px',
                            backgroundColor: getWeightColor(row[outputIdx], weights2Stats.min, weights2Stats.max),
                          }}
                          title={`Caché ${hiddenIdx} → Sortie ${outputIdx}: ${row[outputIdx].toFixed(3)}`}
                        />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <p className="text-xs text-gray-500 mt-1">
              Poids entre les neurones cachés et les neurones de sortie. 
              Chaque colonne représente un chiffre (0-3) et l'intensité du gris indique l'importance de chaque neurone caché pour cette prédiction.
              Blanc = forte influence positive, Noir = forte influence négative.
            </p>
          </div>

          {/* Statistiques d'apprentissage */}
          <div className="bg-gray-50 p-4 rounded-lg col-span-1 md:col-span-2">
            <h3 className="text-lg font-semibold mb-2">Derniers Exemples Appris</h3>
            <div className="overflow-x-auto">
              <table className="min-w-full border-collapse">
                <thead>
                  <tr className="bg-gray-100">
                    <th className="border px-4 py-2">#</th>
                    <th className="border px-4 py-2">Chiffre</th>
                    <th className="border px-4 py-2">Aperçu</th>
                  </tr>
                </thead>
                <tbody>
                  {examples.slice(-5).reverse().map((example, idx) => (
                    <tr key={idx}>
                      <td className="border px-4 py-2 text-center">{examples.length - idx}</td>
                      <td className="border px-4 py-2 text-center font-bold">{example.target}</td>
                      <td className="border px-4 py-2">
                        <div className="grid grid-cols-8 gap-0 mx-auto w-24">
                          {Array.from({ length: 64 }).map((_, i) => (
                            <div 
                              key={i}
                              className={`w-3 h-3 ${example.input[i] > 0 ? 'bg-black' : 'bg-gray-100'} border border-gray-200`}
                            />
                          ))}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Fonction pour obtenir un tableau à partir d'une matrice mathjs
function getMatrixArray(matrix: any): number[][] {
  try {
    if (typeof matrix.toArray === 'function') {
      return matrix.toArray();
    } else if (matrix._data) {
      return matrix._data;
    } else if (Array.isArray(matrix)) {
      return matrix;
    }
  } catch (error) {
    console.error("Error converting matrix to array:", error);
  }
  return [[]];
}

// Calculer les statistiques d'un tableau de nombres
function calculateStats(arr: number[]): { min: number; max: number; mean: number } {
  if (arr.length === 0) return { min: 0, max: 0, mean: 0 };
  
  const min = Math.min(...arr);
  const max = Math.max(...arr);
  const sum = arr.reduce((acc, val) => acc + val, 0);
  const mean = sum / arr.length;
  
  return { min, max, mean };
}

// Générer une couleur en fonction de la valeur du poids (nuances de gris)
function getWeightColor(weight: number, min: number, max: number): string {
  // Normaliser entre -1 et 1
  const range = Math.max(Math.abs(min), Math.abs(max));
  const normalized = range === 0 ? 0 : weight / range;
  
  // Utiliser des nuances de gris: 
  // - Noir pour les valeurs fortement négatives
  // - Blanc pour les valeurs fortement positives
  // - Gris moyen pour les valeurs proches de zéro
  
  // Transformer à l'échelle 0-255 (noir à blanc)
  const intensity = Math.round(((normalized + 1) / 2) * 255);
  return `rgb(${intensity},${intensity},${intensity})`;
}

export default ModelVisualization;