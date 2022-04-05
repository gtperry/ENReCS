using System;
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
        int Q;

        float Learning_rate;
        float L1_penality;
        float L2_penality;
        float B;
        
        float[,] X;
        float[,] Y;
        float[,] W;

        Device dev;
        Accelerator accelerate;
       
        //Will be used later when optimized for GPU use
        Context context;


        //Constructor
        public ElasticRegression(float learning_rate, int iterations, float l1_penality, float l2_penality)
        {
            ///<summary>Constructor for ElasticRegression object</summary>
            ///<param name="learning_rate">(float) learning rate of the regression</param>
            ///<param name="iterations">(int) How many iterations of the algorithm will run when the model is fit</param>
            ///<param name="l1_penality">(float )L1 penality</param>
            ///<param name="l2_penality">(float )L2 penality</param>


            this.context = Context.Create(builder => builder.AllAccelerators());

            this.Learning_rate = learning_rate;

            this.Iterations = iterations;

            this.L1_penality = l1_penality;

            this.L2_penality = l2_penality;

            this.dev = this.context.GetPreferredDevice(preferCPU: false);
            Console.WriteLine(this.dev);

           


        }

        //Function for model training
        public ElasticRegression fit(float[,] X, float[,] Y, bool verbose = true)
        {
            ///<summary>Trains the model</summary>
            ///<param name="X">(float[,]) A 2d array of the inputs to be trained on.</param>
            ///<param name="Y">(float[,]) A 2d array of the target outputs, must have same length as X</param>
            ///<param name="verbose">(boolean) Determines if the program outputs updates as it runs, default = true</param>

            
            //Number of training examples
            this.M = X.GetLength(0)*Y.GetLength(1);
            //Number of features
            this.N = X.GetLength(1);

            //Number of outputs
            this.Q = Y.GetLength(1);
 
            //Initializes variables
            this.W = new float[this.N, this.Q];
            this.B = 0.0f;
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
        public ElasticRegression fitNOGPU(float[,] X, float[,] Y, bool verbose = true)
        {
            ///<summary>Trains the model</summary>
            ///<param name="X">(float[,]) A 2d array of the inputs to be trained on.</param>
            ///<param name="Y">(float[,]) A 2d array of the target outputs, must have same length as X</param>
            ///<param name="verbose">(boolean) Determines if the program outputs updates as it runs, default = true</param>

            
            //Number of training examples
            this.M = X.GetLength(0)*Y.GetLength(1);
            //Number of features
            this.N = X.GetLength(1);

            //Number of outputs
            this.Q = Y.GetLength(1);
 
            //Initializes variables
            this.W = new float[this.N, this.Q];
            this.B = 0.0f;
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
                this.update_weightsNOGPU();
            }

            return this;
        }

        //Helper function to update weights in gradient descent
        public ElasticRegression update_weights()
        {
            //Generate a prediction based on inputs
            float[,] Y_pred = this.predict(this.X);
            // Console.WriteLine("Y PRED HERE ");
            // print2d(Y_pred);
            // Console.WriteLine("");
            // Console.WriteLine("");
            // Console.WriteLine("");
            // Console.WriteLine("-----------------");
           
            this.accelerate = this.dev.CreateAccelerator(this.context);

            //calculate gradients  
            float[,] dW = new float[this.N, this.Q];
            for(int j = 0; j < this.N; j++)
            {
                for(int z = 0; z < this.Q; z++){
                    if (this.W[j, z] > 0)
                    {
                        dW[j, z] = (((-(MatrixMultiplyAccelerated(this.accelerate, subtwoarrs(this.Y, Y_pred), this.X))[z, j] * (2.0f + this.L1_penality))) + (2 * this.L2_penality * this.W[j, z])) / this.M; ;
                    }
                    else
                    {
                        dW[j, z] = (((-(MatrixMultiplyAccelerated(this.accelerate, subtwoarrs(this.Y, Y_pred), this.X))[z, j] * (2.0f - this.L1_penality))) + (2 * this.L2_penality * this.W[j, z])) / this.M; ;
                    }
                }
                
            }

            float db = (-(2.0F * ysum(subtwoarrs(this.Y, Y_pred)))) / this.M;
            this.W = subtwo2darrs(this.W, applymul(dW, this.Learning_rate));
            this.B = this.B - (this.Learning_rate * db);

            this.accelerate.Dispose();
            return this;
        }

         public ElasticRegression update_weightsNOGPU()
        {
            //Generate a prediction based on inputs
            float[,] Y_pred = this.predictNOGPU(this.X);
            // Console.WriteLine("Y PRED HERE ");
            // print2d(Y_pred);
            // Console.WriteLine("");
            // Console.WriteLine("");
            // Console.WriteLine("");
            // Console.WriteLine("-----------------");
           
            

            //calculate gradients  
            float[,] dW = new float[this.N, this.Q];
            for(int j = 0; j < this.N; j++)
            {
                for(int z = 0; z < this.Q; z++){
                    if (this.W[j, z] > 0)
                    {
                        dW[j, z] = (((-(matrixmul(subtwoarrs(this.Y, Y_pred), this.X))[z, j] * (2.0f + this.L1_penality))) + (2 * this.L2_penality * this.W[j, z])) / this.M; ;
                    }
                    else
                    {
                        dW[j, z] = (((-(matrixmul(subtwoarrs(this.Y, Y_pred), this.X))[z, j] * (2.0f - this.L1_penality))) + (2 * this.L2_penality * this.W[j, z])) / this.M; ;
                    }
                }
                
            }

            float db = (-(2.0F * ysum(subtwoarrs(this.Y, Y_pred)))) / this.M;
            this.W = subtwo2darrs(this.W, applymul(dW, this.Learning_rate));
            this.B = this.B - (this.Learning_rate * db);

            
            return this;
        }

        
        //Matrix Multiplication (dot product)
        float[,] matrixmul(float[,] x, float[,] y)
        {
            ///<summary>Does matrix multiplication on two 2d arrays</summary>
            ///<param name="x">Array 1</param>
            ///<param name="y">Array 2</param>


            //Initialize all varaibles
            int m = x.GetLength(0), n = x.GetLength(1), p = y.GetLength(0), q = y.GetLength(1), i, j;

            //Create empty array of new size
            float[,] c = new float[m, q];

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

        static float[,] MatrixMultiplyAccelerated(Accelerator accelerator, float[,] a, float[,] b)
        {
            var m = a.GetLength(0);
            var ka = a.GetLength(1);
            var kb = b.GetLength(0);
            var n = b.GetLength(1);

            if (ka != kb)
                throw new ArgumentException($"Cannot multiply {m}x{ka} matrix by {n}x{kb} matrix", nameof(b));

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(
                MatrixMultiplyAcceleratedKernel);

            using var aBuffer = accelerator.Allocate2DDenseX<float>(new Index2D(m, ka));
            using var bBuffer = accelerator.Allocate2DDenseX<float>(new Index2D(ka, n));
            using var cBuffer = accelerator.Allocate2DDenseX<float>(new Index2D(m, n));
            aBuffer.CopyFromCPU(a);
            bBuffer.CopyFromCPU(b);

            kernel(cBuffer.Extent.ToIntIndex(), aBuffer.View, bBuffer.View, cBuffer.View);

            // Reads data from the GPU buffer into a new CPU array.
            // Implicitly calls accelerator.DefaultStream.Synchronize() to ensure
            // that the kernel and memory copy are completed first.
            return cBuffer.GetAsArray2D();
        }


        static void MatrixMultiplyAcceleratedKernel(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> aView,
            ArrayView2D<float, Stride2D.DenseX> bView,
            ArrayView2D<float, Stride2D.DenseX> cView)
        {
            var x = index.X;
            var y = index.Y;
            var sum = 0.0f;

            for (var i = 0; i < aView.IntExtent.Y; i++)
                sum += aView[new Index2D(x, i)] * bView[new Index2D(i, y)];

            cView[index] = sum;
        }

        //Predicts outputs based off of x

        float[,] predict(float[,] x)
            ///<summary>Predicts output based off of x</summary>
            ///<param name="x">Array of inputs</param>
        { 
            this.accelerate = this.dev.CreateAccelerator(this.context);
            float[,] prediction = applyadd(MatrixMultiplyAccelerated(this.accelerate, x, this.W), this.B);
            this.accelerate.Dispose();
            return prediction;
        }
        float[,] predictNOGPU(float[,] x)
            ///<summary>Predicts output based off of x</summary>
            ///<param name="x">Array of inputs</param>
        { 
            
           return applyadd(matrixmul(x, this.W), this.B);
            
        }

        //Adds a value to each member of a 2d array
        float[,] applyadd(float[,] arr, float val)
            ///<summary>Adds a value to each member of a 2d array</summary>
        {
            float[,] temp = new float[arr.GetLength(0), arr.GetLength(1)];
            for (int i = 0; i < arr.GetLength(0); i++)
            {
                for (int j = 0; j < arr.GetLength(1); j++)
                {
                    temp[i, j] = arr[i, j] + val;
                }
            }
            return temp;

        }
        //Multiplies each member of a 2d array by a value
        float[,] applymul(float[,] arr, float val)
            ///<summary>Multiplies each member of a 2d array by a value</summary>
        {
            float[,] temp = new float[arr.GetLength(0), arr.GetLength(1)];
            for (int i = 0; i < arr.GetLength(0); i++)
            {
                for (int j = 0; j < arr.GetLength(1); j++)
                {
                    temp[i, j] = arr[i, j] * val;
                }
            }
            return temp;
        }
        
        
        float[,] subtwoarrs(float[,] arr, float[,] val)
            ///<summary>subtracts the values of an array from another one, and returns the results, with the rows and columns switched</summary>
        {
            float[,] temp = new float[arr.GetLength(1), arr.GetLength(0)];
            for (int i = 0; i < arr.GetLength(0); i++)
            {
                for (int j = 0; j < arr.GetLength(1); j++)
                {
                    temp[j, i] = arr[i, j] - val[i, j];
                }
            }
            return temp;

        }
        
        float[,] subtwo2darrs(float[,] array1, float[,] array2)
            ///<summary>Subtracts the values of arr2 from the corresponding ones in array1. Arrays must be same dimensions</summary>
            
        {
            float[,] temp = new float[array1.GetLength(0), array1.GetLength(1)];
            for (int i = 0; i < array1.GetLength(0); i++)
            {
                for (int j = 0; j < array1.GetLength(1); j++)
                {
                    temp[i, j] = array1[i, j] - array2[i, j];
                }
            }
            return temp;
        }

        //HelperFunction that gets the sum of Y_pred or this.Y
        //***This will need to be changed to accept multiple outputs
        float ysum(float[,] y)
        {
            float total = 0;
            for (int i = 0; i < y.GetLength(1); i++)
            {
                total = total + y[0, i];
            }
            return total;
        }

        //Helper function used for testing, prints 1d array
        void print1d(float[] array)
        {

            for (int j = 0; j < array.GetLength(0); j++)
            {
                Console.Write("{0} ", array[j]);
            }

        }
        //Helper function used for testing, prints 2d array
        void print2d(float[,] array)
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

        static void Main(string[] args)
        {
            // //learning_rate, iterations, l1_penality, l2_penality 
            ElasticRegression e1 = new ElasticRegression(0.005f, 10, .05f, .05f);
            ElasticRegression e2 = new ElasticRegression(0.005f, 10, .05f, .05f);
     
            Random q = new Random();
            //Creates input data
            float[,] Xactual = new float[10000, 50];
            for (int i = 0; i < Xactual.GetLength(0); i++)
            {
                for(int j=0; j < Xactual.GetLength(1); j++){
                    Xactual[i, j] = ((float)q.NextDouble() * 10) + 2;
                }
                //Xactual[i, 1] = ((float)q.NextDouble() * 10) + 2;
               
            }

            //Creates output data
            float[,] Yactual = new float[10000, 50];
            for (int i = 0; i < Xactual.GetLength(0); i++)
            {
                for(int j = 0; i < Xactual.GetLength(1); i++){
                    Yactual[i, j] = ((Xactual[i, j]) * 100 + 2000);
                }
                // Yactual[i, 0] = ((Xactual[i, 0]) * 1000 + 2000);
                // Yactual[i, 1] = ((Xactual[i, 0]) * 100 + 100);
                //Yactual[i, 1] = ((Xactual[i, 0]) * 1000 + 2000);
            }
            
            
           
            // Console.WriteLine("Finshed Building Data");
            
            
        
            // // var context = Context.CreateDefault();
            // // Stopwatch stopwatch = new Stopwatch();

            // // Accelerator accelerator = context.CreateCPUAccelerator(0);
            
            // //First tests at using accelerator, no improvement on runtime.
            // //stopwatch.Start();
            // //using (accelerator)
            // //{
            // //    Console.WriteLine("Before fit");
            // //    e1.fit(Xactual, Yactual);
            // //}
            // //accelerator.Dispose();
            // //stopwatch.Stop();
            // //Console.WriteLine("With accelerator");
            // //Console.WriteLine(stopwatch.Elapsed);


             
            // // Stopwatch stopwatch2 = new Stopwatch();


            // // stopwatch2.Start();

            // Console.WriteLine("Before fit");
            Stopwatch stopw1 = new Stopwatch();
            stopw1.Start();
            e1.fit(Xactual, Yactual);
            stopw1.Stop();

            Stopwatch stopw2 = new Stopwatch();
            stopw2.Start();
            e2.fitNOGPU(Xactual, Yactual);
            stopw2.Stop();

            // // stopwatch2.Stop();
            // Console.WriteLine("Without accelerator");
            // //


            Stopwatch stopw = new Stopwatch();
            stopw.Start();
            float[,] res = e1.predict(Xactual);
            stopw.Stop();

            Stopwatch stopw3 = new Stopwatch();
            stopw3.Start();
            float[,] res2 = e2.predictNOGPU(Xactual);
            stopw3.Stop();


            //This prints out the prediction with the actual
            // for (int i = 0; i < res.GetLength(0); i++)
            // {
            //    Console.Write(res[i, 0]);
            //    Console.Write(" ");
            //    Console.Write(res[i, 1]);
            //    Console.Write(" | ");
            //    Console.Write(Yactual[i, 0]);
            //    Console.Write(" ");
            //    Console.Write(Yactual[i, 1]);

            //    Console.WriteLine();
            // }

            Console.WriteLine("With GPU:");
            Console.WriteLine(stopw1.Elapsed);
            Console.WriteLine(stopw.Elapsed);

            Console.WriteLine("Without GPU:");
            Console.WriteLine(stopw2.Elapsed);
            Console.WriteLine(stopw3.Elapsed);
            // Console.WriteLine("Done");
        
   
            // Context context = Context.Create(builder => builder.AllAccelerators());

            // foreach (Device device in context)
            // {
            //     Console.WriteLine(device);
            // }
            // using Context context = Context.Create(builder => builder.AllAccelerators());
            // Console.WriteLine("Context: " + context.ToString());

            // Device d = context.GetPreferredDevice(preferCPU: false);
            // Accelerator a = d.CreateAccelerator(context);

            // a.PrintInformation();
            // a.Dispose();

            // foreach(Device device in context.GetPreferredDevices(preferCPU: false, matchingDevicesOnly: false))
            // {
            //     Accelerator accelerator = device.CreateAccelerator(context);
            //     accelerator.PrintInformation();
            //     accelerator.Dispose();
            // }
        }

        
    }
}
