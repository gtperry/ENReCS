﻿using System;
using System.IO;
using System.Threading.Tasks;
using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;

namespace ElasticNetRegression
{
    public class ElasticRegression
    {
        int Iterations;
        int M;
        int N;

        double Learning_rate;
        double L1_penality;
        double L2_penality;
        double B;
        
        double[,] X;
        double[,] Y;
        double[,] W;

        //Will be used later when optimized for GPU use
        //Context context;


        //Constructor
        public ElasticRegression(double learning_rate, int iterations, double l1_penality, double l2_penality)
        {
            ///<summary>Constructor for ElasticRegression object</summary>
            ///<param name="learning_rate">(double) learning rate of the regression</param>
            ///<param name="iterations">(int) How many iterations of the algorithm will run when the model is fit</param>
            ///<param name="l1_penality">(double )L1 penality</param>
            ///<param name="l2_penality">(double )L2 penality</param>

            this.Learning_rate = learning_rate;

            this.Iterations = iterations;

            this.L1_penality = l1_penality;

            this.L2_penality = l2_penality;
        }

        //Function for model training
        public ElasticRegression fit(double[,] X, double[,] Y, bool verbose = true)
        {
            ///<summary>Trains the model</summary>
            ///<param name="X">(double[,]) A 2d array of the inputs to be trained on.</param>
            ///<param name="Y">(double[,]) A 2d array of the target outputs, must have same length as X</param>
            ///<param name="verbose">(boolean) Determines if the program outputs updates as it runs, default = true</param>

            
            //Number of training examples
            this.M = X.GetLength(0);
            //Number of features
            this.N = X.GetLength(1);

            //Initializes variables
            this.W = new double[this.N, 1];
            this.B = 0.0;
            this.X = X;
            this.Y = Y;

            //Gradient descent learning
            for(int i = 0; i < this.Iterations; i++)
            {
                if (verbose)
                {
                    Console.WriteLine("Iteration {0}/{1}", i, this.Iterations);
                }

                //Updates the weights after each iteration
                this.update_weights();
            }

            return this;
        }

        //Helper function to update weights in gradient descent
        public ElasticRegression update_weights()
        {
            //Generate a prediction based on inputs
            double[,] Y_pred = this.predict(this.X);
           
            //calculate gradients  
            double[,] dW = new double[this.N, 1];
            for(int j = 0; j < this.N; j++)
            {
                if (this.W[j, 0] > 0)
                {
                    dW[j, 0] = (((-(matrixmul(subtwoarrs(this.Y, Y_pred), this.X))[0, j] * 2.0) + this.L1_penality) + (2 * this.L2_penality * this.W[j, 0])) / this.M; ;
                }
                else
                {
                    dW[j, 0] = (((-(matrixmul(subtwoarrs(this.Y, Y_pred), this.X))[0, j] * 2.0) - this.L1_penality) + (2 * this.L2_penality * this.W[j, 0])) / this.M; ;
                }
            }
            double db = (-(2.0 * ysum(subtwoarrs(this.Y, Y_pred)))) / this.M;
            this.W = subtwo2darrs(this.W, applymul(dW, this.Learning_rate));
            this.B = this.B - this.Learning_rate * db;
            return this;
        }

        
        //Matrix Multiplication (dot product)
        double[,] matrixmul(double[,] x, double[,] y)
        {
            ///<summary>Does matrix multiplication on two 2d arrays</summary>
            ///<param name="x">Array 1</param>
            ///<param name="y">Array 2</param>
           
            //Initialize all varaibles
            int m = x.GetLength(0), n = x.GetLength(1), p = y.GetLength(0), q = y.GetLength(1), i, j;

            //Create empty array of new size
            double[,] c = new double[m, q];

            //Check that the arrays are the correct sizes, and compatible
            if (n != p)
            {
                Console.WriteLine("Matrix multiplication not possible");
            }
            else
            {
                for (i = 0; i < m; i++)
                {
                    for (j = 0; j < q; j++)
                    {
                        c[i, j] = 0;
                        for (int k = 0; k < n; k++)
                        {
                            c[i, j] += x[i, k] * y[k, j];
                        }
                    }
                }
            }
            return c;
        }
        double[,] predict(double[,] x)
        {
           return applyadd(matrixmul(x, this.W), this.B);
        }
        //Helper function used for testing
        void print1d(double[] array)
        {

            for (int j = 0; j < array.GetLength(0); j++)
            {
                Console.Write("{0} ", array[j]);
            }

        }
        //Helper function used for testing
        void print2d(double[,] array)
        {
            Console.Write("[");
            for (int i = 0; i < array.GetLength(0); i++)
            {
                Console.Write("[");
                for (int j = 0; j < array.GetLength(1); j++)
                {
                    Console.Write("{0}, ", array[i, j]);
                }
                Console.Write("]");
                Console.Write(", ");
            }
            Console.WriteLine("]");
        }
        double[,] applyadd(double[,] arr, double val)
        {
            double[,] temp = new double[arr.GetLength(0), arr.GetLength(1)];
            for (int i = 0; i < arr.GetLength(0); i++)
            {
                for (int j = 0; j < arr.GetLength(1); j++)
                {
                    temp[i, j] = arr[i, j] + val;
                }
            }
            return temp;

        }
        
        double[,] applymul(double[,] arr, double val)
        {
            double[,] temp = new double[arr.GetLength(0), arr.GetLength(1)];
            for (int i = 0; i < arr.GetLength(0); i++)
            {
                for (int j = 0; j < arr.GetLength(1); j++)
                {
                    temp[i, j] = arr[i, j] * val;
                }
            }
            return temp;

        }
        
        double[,] subtwoarrs(double[,] arr, double[,] val)
        {
            double[,] temp = new double[arr.GetLength(1), arr.GetLength(0)];
            for (int i = 0; i < arr.GetLength(0); i++)
            {
                for (int j = 0; j < arr.GetLength(1); j++)
                {
                    temp[j, i] = arr[i, j] - val[i, j];
                }
            }
            return temp;

        }
        
        double[,] subtwo2darrs(double[,] array1, double[,] array2)
            ///<summary>Subtracts the values of arr2 from the corresponding ones in array1. Arrays must be same dimensions</summary>
            
        {
            //double[,] temp = new double[array1.GetLength(0), array1.GetLength(1)];
            for (int i = 0; i < array1.GetLength(0); i++)
            {
                for (int j = 0; j < array1.GetLength(1); j++)
                {
                    array1[i, j] = array1[i, j] - array2[i, j];
                }
            }
            return array1;
        }

        //HelperFunction that gets the sum of Y_pred or this.Y
        //***This will need to be changed to accept multiple outputs
        double ysum(double[,] y)
        {
            double total = 0;
            for (int i = 0; i < y.GetLength(1); i++)
            {
                total = total + y[0, i];
            }
            return total;
        }

        static void Main(string[] args)
        {
            //learning_rate, iterations, l1_penality, l2_penality 
            ElasticRegression e1 = new ElasticRegression(0.01, 100, .5, .5);
            ElasticRegression e2 = new ElasticRegression(0.01, 100, .5, .5);
     
            Random q = new Random();
            double[,] Xactual = new double[10000, 1];
            for (int i = 0; i < Xactual.GetLength(0); i++)
            {
                Xactual[i, 0] = (q.NextDouble() * 100) + 2;
                //Xactual[i, 1] = (q.NextDouble() * 10) + 2;
            }
            Console.WriteLine("XBuilt");
            double[,] Yactual = new double[10000, 1];
            for (int i = 0; i < Xactual.GetLength(0); i++)
            {
                Yactual[i, 0] = ((Xactual[i, 0])* (Xactual[i, 0]) * 100 + 20000);
            }
            //e.print2d();
            
            
            Console.WriteLine("FinshedBuilding");
            
            
            Console.WriteLine("Final Prediction:");
            //e.print2d(e.predict(Xactual));
            var context = Context.CreateDefault();
            Stopwatch stopwatch = new Stopwatch();

            Accelerator accelerator = context.CreateCPUAccelerator(0);

            stopwatch.Start();
            using (accelerator)
            {
                Console.WriteLine("Before fit");
                e1.fit(Xactual, Yactual);
            }
            accelerator.Dispose();
            stopwatch.Stop();
            Console.WriteLine("With accelerator");
            Console.WriteLine(stopwatch.Elapsed);

            Stopwatch stopwatch2 = new Stopwatch();

            
            stopwatch2.Start();
           
            Console.WriteLine("Before fit");
            e2.fit(Xactual, Yactual);
            
            stopwatch2.Stop();
            Console.WriteLine("Withpout accelerator");
            Console.WriteLine(stopwatch2.Elapsed);


            foreach (CPUDevice qqq in context.GetCPUDevices())
            {
                using CPUAccelerator accelerator2 = (CPUAccelerator)qqq.CreateAccelerator(context);
                Console.WriteLine(accelerator2);
                //Console.WriteLine(GetInfoString(accelerator2));
            }
            double[,] res = e1.predict(Xactual);
            //Console.WriteLine("Prediction:");
            //e.print2d(res);
            //Console.WriteLine("");
            //Console.WriteLine("Actual:");
            //e.print2d(Yactual);
            for (int i = 0; i < res.GetLength(0); i++)
            {
                Console.Write(res[i, 0]);
                Console.Write(", ");
                Console.Write(Yactual[i, 0]);
                Console.WriteLine();
            }

            Console.WriteLine("Done");

        }
    }

}