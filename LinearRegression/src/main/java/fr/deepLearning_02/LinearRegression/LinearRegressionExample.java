package fr.deepLearning_02.LinearRegression;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class LinearRegressionExample {
	public static void main(String[] args) {
		// Définir un réseau neuronal simple avec une seule couche
		MultiLayerNetwork model = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
				.seed(123)
				.weightInit(WeightInit.XAVIER)
				.updater(new Nesterovs(0.01))	// Utilisation correcte du taux d'apprentissage
				.list()
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)	// Mean Squared Error
						.activation(Activation.IDENTITY)	//activation linéaire
						.nIn(1)
						.nOut(1)
						.build())
				.build());
		
		model.init();
		
		// Données d'entraînement : x = {1, 2, 3, 4}, y = {3, 5, 7, 9}
        // Correspond à y = 2 x + 1
		INDArray input = Nd4j.create(new double[] {1, 2, 3, 4}, new int[] {4, 1});
		INDArray labels = Nd4j.create(new double[] {3, 5, 7, 9}, new int[] {4, 1});
		
		DataSet dataSet = new DataSet(input, labels);
		
		//Entraînement
		for(int i = 0; i < 2000; i++) {
			model.fit(dataSet);
		    if (i % 500 == 0) {
		        System.out.println("Époque " + i + " - Score: " + model.score());
		    }
		}
		
		//Faire une prédiction
		INDArray testinput = Nd4j.create(new double[] {10, 100}, new int[] {2, 1});
		INDArray output = model.output(testinput);
		
		INDArray weights = model.getParam("0_W");
		INDArray bias = model.getParam("0_b");
		
		// Afficher les résultats
        System.out.println("Prédiction pour x=10 : " + output.getDouble(0));
        System.out.println("Prédiction pour x=100 : " + output.getDouble(1));
        System.out.println("Poids entraîné : " + weights.getDouble(0));
        System.out.println("Biais entraîné : " + bias.getDouble(0));
        
        model.close();
	}
}
