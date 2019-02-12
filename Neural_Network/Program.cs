using System;

namespace Neural_Network
{
    internal class NuralNet
    {
        public static void Main()
        {
            double w1, w2, b, learning_rate = 0.0;
            int iterations = 0;

            // ------------------------------------------------------------------------------------------------------------------------------------------
            // Fetch Data from User & feed Test Data
            // ------------------------------------------------------------------------------------------------------------------------------------------            
            double[][] arrSuperset = feedTestingData();
            double[] queryData = getUserQuery();
            learning_rate = double.Parse(getUserData("Learning Rate"));
            iterations = int.Parse(getUserData("Iterations"));

            // ------------------------------------------------------------------------------------------------------------------------------------------
            // Generate Random Values for Weights
            // ------------------------------------------------------------------------------------------------------------------------------------------
            w1 = getRandomRumber(arrSuperset);
            w2 = getRandomRumber(arrSuperset);
            b = getRandomRumber(arrSuperset);

            // ------------------------------------------------------------------------------------------------------------------------------------------
            // Start training Loop
            // ------------------------------------------------------------------------------------------------------------------------------------------
            Console.WriteLine("Starting training Loop..");
            double[] arrWeights = trainingLoop(iterations, arrSuperset, learning_rate, w1, w2, b);

            // ------------------------------------------------------------------------------------------------------------------------------------------
            // Calculate result based on finalized weights 
            // ------------------------------------------------------------------------------------------------------------------------------------------
            double result = calculateAfterMath(arrWeights, queryData);

            if (result == 0)
            {
                Console.WriteLine("It is Blue !!");
            }
            else
            {
                Console.WriteLine("It is Red !!");
            }

            Console.WriteLine("Neural Network is ready..");
            Console.Read();
        }

        public static double calculateAfterMath(double[] arrWeights, double[] queryData)
        {
            double w1 = arrWeights[0];
            double w2 = arrWeights[1];
            double b = arrWeights[2];

            // ------------------------------------------------------------------------------------------------------------------------------------------
            // Calculate result by modified weights 
            // ------------------------------------------------------------------------------------------------------------------------------------------
            int result = (int)Math.Round((w1 * queryData[0]) + (w2 * queryData[1]) + b);

            Console.WriteLine("W1 : " + w1);
            Console.WriteLine("W2 : " + w2);
            Console.WriteLine("b : " + b);
            Console.WriteLine("------------------------------------------");

            result = (int)Math.Round(sigmoid(result));
            Console.WriteLine("Result Value : " + result);

            return result;
        }

        public static double[] trainingLoop(int iterations, double[][] arrSuperset, double learning_rate, double w1, double w2, double b)
        {
            Random generator2 = new Random();
            double[] currentData;
            double len, wid, target, pred, z, cost, dCOSTdX, dPREDdX, dZdW1, dZdW2, dZdB, dCOSTdW1, dCOSTdW2, dCOSTdB;
            // ------------------------------------------------------------------------------------------------------------------------------------------
            // LOOP: Training loop 
            // ------------------------------------------------------------------------------------------------------------------------------------------
            for (int index = 0; index < iterations; index++)
            {
                // ------------------------------------------------------------------------------------------------------------------------------------------
                // Fetch Random dataset from array of dataset 
                // ------------------------------------------------------------------------------------------------------------------------------------------
                currentData = arrSuperset[generator2.Next(0, arrSuperset.Length)];
                len = currentData[0];
                wid = currentData[1];
                target = currentData[2];

                // ------------------------------------------------------------------------------------------------------------------------------------------
                // Calculate prediction from dataset
                // ------------------------------------------------------------------------------------------------------------------------------------------
                #region Calculate Prediction

                z = w1 * len + w2 * wid + b;
                pred = sigmoid(z);

                #endregion Calculate Prediction

                // ------------------------------------------------------------------------------------------------------------------------------------------
                // Calculate cost based on prediction
                // ------------------------------------------------------------------------------------------------------------------------------------------
                #region Calculate Cost

                cost = (pred - target) * (pred - target);

                #endregion Calculate Cost

                // ------------------------------------------------------------------------------------------------------------------------------------------
                // Calculate Partial derivative of cost with respect to all weights
                // ------------------------------------------------------------------------------------------------------------------------------------------
                #region Calculate Derivative of Cost WRT. all Variables

                dCOSTdX = 2 * (pred - target);

                dPREDdX = sigmoid(z) * (1 - sigmoid(z));

                dZdW1 = len;
                dZdW2 = wid;
                dZdB = 1;

                dCOSTdW1 = dCOSTdX * dPREDdX * dZdW1;
                dCOSTdW2 = dCOSTdX * dPREDdX * dZdW2;
                dCOSTdB = dCOSTdX * dPREDdX * dZdB;

                #endregion Calculate Derivative of Cost WRT. all Variables

                // ------------------------------------------------------------------------------------------------------------------------------------------
                // Substract the fraction of derivative from the original weights to move them towards stable state 
                // ------------------------------------------------------------------------------------------------------------------------------------------
                #region Substract fraction of derivative from each weights

                w1 -= learning_rate * dCOSTdW1;
                w2 -= learning_rate * dCOSTdW2;
                b -= learning_rate * dCOSTdB;

                #endregion Substract fraction of derivative from each weights
            }

            double[] arrWeights = new double[] { w1, w2, b };
            return arrWeights;
        }

        public static double sigmoid(double value)
        {
            // ------------------------------------------------------------------------------------------------------------------------------------------
            // Calculate sigmoid of value ( convert betbeen 0 - 1) 
            // ------------------------------------------------------------------------------------------------------------------------------------------
            return 1 / (1 + Math.Exp(-value));
        }

        // ------------------------------------------------------------------------------------------------------------------------------------------
        // UTILITY FUNCTIONS 
        // ------------------------------------------------------------------------------------------------------------------------------------------

        public static double getRandomRumber(double[][] arrSuperset)
        {
            double minData = arrSuperset[0][0];
            double maxData = arrSuperset[0][1];
            double[] data;
            for (int indx = 1; indx < arrSuperset.Length; indx++)
            {
                data = arrSuperset[indx];
                if (data[0] < minData)
                {
                    minData = data[0];
                }
                if (data[1] > maxData)
                {
                    maxData = data[1];
                }
            }

            Random generator = new Random();

            double randomValue = generator.Next((int)minData, (int)maxData);
            return randomValue;
        }

        public static double[][] feedTestingData()
        {
            double[] dataB1 = new double[] { 1, 1, 0 };
            double[] dataB2 = new double[] { 2, 1, 0 };
            double[] dataB3 = new double[] { 2, 0.5, 0 };
            double[] dataB4 = new double[] { 3, 1, 0 };
            double[] dataR1 = new double[] { 3, 1.5, 1 };
            double[] dataR2 = new double[] { 3.5, 0.5, 1 };
            double[] dataR3 = new double[] { 4, 1.5, 1 };
            double[] dataR4 = new double[] { 5.5, 1, 1 };
            double[][] arrSuperset = new double[8][] { dataB1, dataB2, dataB3, dataB4, dataR1, dataR2, dataR3, dataR4 };
            return arrSuperset;
        }

        public static double[] getUserQuery()
        {
            Console.WriteLine("Enter the value of parameters:");
            double[] arrQuery = new double[3];
            arrQuery[0] = double.Parse(Console.ReadLine());
            arrQuery[1] = double.Parse(Console.ReadLine());
            arrQuery[2] = 0;
            return arrQuery;
        }

        public static string getUserData(string dataToBeFetched)
        {
            Console.WriteLine("Enter " + dataToBeFetched + ": ");
            return Console.ReadLine();
        }
    }
}