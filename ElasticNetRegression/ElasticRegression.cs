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
        int numfeatures;

        float Learning_rate;
        float L1_penality;
        float L2_penality;
        float B;
        
        float[,] X;
        float[,] Y;
        float[,] W;
        float[,] multipliedMatrix;
        

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
            this.M = X.GetLength(0)*X.GetLength(1);
            //Number of features
            this.N = X.GetLength(1);
            this.numfeatures = X.GetLength(1);
            
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
        public ElasticRegression fitTILED(float[,] X, float[,] Y, bool verbose = true)
        {
            ///<summary>Trains the model</summary>
            ///<param name="X">(float[,]) A 2d array of the inputs to be trained on.</param>
            ///<param name="Y">(float[,]) A 2d array of the target outputs, must have same length as X</param>
            ///<param name="verbose">(boolean) Determines if the program outputs updates as it runs, default = true</param>

            
            //Number of training examples
            this.M = X.GetLength(0)*X.GetLength(1);
            //Number of features
            this.N = X.GetLength(1);
            this.numfeatures = X.GetLength(1);
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
                this.update_weightsTILED();
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
            this.M = X.GetLength(0)*X.GetLength(1);
            //Number of features
            this.N = X.GetLength(1);
            this.numfeatures = X.GetLength(1);
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
            //Console.WriteLine("X");
            //this.print2d(this.X);
            //Console.WriteLine("subtwoarrs");
            //this.print2d(subtwoarrs(this.Y, Y_pred));
            // Console.WriteLine("Going step by step to find growth");
            // Console.WriteLine("Y_pred");
            // this.print2d(Y_pred);
            // Console.ReadLine();
            // Console.WriteLine("Y_Actual");
            // this.print2d(this.Y);
            // Console.ReadLine();
            // Console.WriteLine("YDiff");
            // this.print2d(subtwoarrs(this.Y, Y_pred));
            // Console.ReadLine();
            // Console.WriteLine("X");
            // this.print2d(this.X);
            // Console.ReadLine();
            // Console.WriteLine("Matrix Multiplied");
            // this.print2d(MatrixMultiplyAccelerated(this.accelerate, subtwoarrs(this.Y, Y_pred), this.X));
            // Console.WriteLine("M");
            // Console.WriteLine(this.M);
            // Console.ReadLine();









            //Multiplied matrix has # of Youtputs rows and # of X features columns
            this.multipliedMatrix = (MatrixMultiplyAccelerated(this.accelerate, subtwoarrs(this.Y, Y_pred), this.X));
            //float[,] multipliedMatrix2 = (MatrixMultiplyTiled(this.accelerate, subtwoarrs(this.Y, Y_pred), this.X));
            // Console.WriteLine("mULT MNATRIX");
            // this.print2d(this.multipliedMatrix);
            // Console.WriteLine("W");
            // this.print2d(this.W);
            // Console.ReadLine();

            float[,] dW = new float[this.N, this.Q];
            for(int j = 0; j < this.N; j++)
            {
                for(int z = 0; z < this.Q; z++){
                    if (this.W[j, z] > 0)
                    {




                        dW[j, z] =  (((-1* this.multipliedMatrix[z, j] * (2.0f + this.L1_penality))) + ( this.L2_penality * this.W[j, z])) / this.M; //(((-this.multipliedMatrix[z, j] * (2.0f + this.L1_penality))) + (2 * this.L2_penality * this.W[j, z])) / this.M; ;
                        // Console.WriteLine("CompA:");
                        // Console.WriteLine(this.multipliedMatrix[z,j]);
                        // Console.WriteLine("Base:");
                        // Console.WriteLine((((-this.multipliedMatrix[z, j] * (2.0f + this.L1_penality))) + (2 * this.L2_penality * this.W[j, z])) / this.M);
                        // Console.WriteLine("Four1:");
                        // Console.WriteLine((((-this.multipliedMatrix[z, j] * (4.0f + this.L1_penality))) + (2 * this.L2_penality * this.W[j, z])) / this.M);
                        // Console.WriteLine("Four2");
                        // Console.WriteLine((((-this.multipliedMatrix[z, j] * (2.0f + this.L1_penality))) + (4 * this.L2_penality * this.W[j, z])) / this.M);
                        // Console.WriteLine("Num Feature1:");
                        // Console.WriteLine((((-this.multipliedMatrix[z, j] * (this.numfeatures + this.L1_penality))) + (2 * this.L2_penality * this.W[j, z])) / this.M);
                        // Console.WriteLine("Num Feature2:");
                        // Console.WriteLine((((-this.multipliedMatrix[z, j] * (2.0f + this.L1_penality))) + (this.numfeatures * this.L2_penality * this.W[j, z])) / this.M);
                        // Console.WriteLine("Minus");
                        // Console.WriteLine((((-this.multipliedMatrix[z, j] * (2.0f + this.L1_penality))) - (2 * this.L2_penality * this.W[j, z])) / this.M);
                        // Console.WriteLine("No Double");
                        // Console.WriteLine((((-this.multipliedMatrix[z, j] * this.L1_penality)) - (2 * this.L2_penality * this.W[j, z])) / this.M);
                        // Console.WriteLine("Div:");
                        // Console.WriteLine((((-this.multipliedMatrix[z, j]/this.M * (2.0f + this.L1_penality))) + (2 * this.L2_penality * this.W[j, z])) / this.M);
                        // Console.WriteLine("MISC");
                        // Console.WriteLine((((-this.multipliedMatrix[z, j] * this.L1_penality)) + (2 * this.L2_penality * this.W[j, z])) / this.M);
                        //Console.ReadLine();
                    }
                    else
                    {
                        dW[j, z] =  (((-1 * this.multipliedMatrix[z, j] * (2.0f - this.L1_penality))) + ( this.L2_penality * this.W[j, z])) / this.M; //(((-this.multipliedMatrix[z, j] * (2.0f - this.L1_penality))) + (2 * this.L2_penality * this.W[j, z])) / this.M; ; 
                        // Console.WriteLine("CompB:");
                        // Console.WriteLine(this.multipliedMatrix[z,j]);
                        // Console.WriteLine("Base:");
                        // Console.WriteLine((((-this.multipliedMatrix[z, j] * (2.0f + this.L1_penality))) + (2 * this.L2_penality * this.W[j, z])) / this.M);
                        // Console.WriteLine("Four1:");
                        // Console.WriteLine((((-this.multipliedMatrix[z, j] * (4.0f + this.L1_penality))) + (2 * this.L2_penality * this.W[j, z])) / this.M);
                        // Console.WriteLine("Four2");
                        // Console.WriteLine((((-this.multipliedMatrix[z, j] * (2.0f + this.L1_penality))) + (4 * this.L2_penality * this.W[j, z])) / this.M);
                        // Console.WriteLine("Num Feature1:");
                        // Console.WriteLine((((-this.multipliedMatrix[z, j] * (this.numfeatures + this.L1_penality))) + (2 * this.L2_penality * this.W[j, z])) / this.M);
                        // Console.WriteLine("Num Feature2:");
                        // Console.WriteLine((((-this.multipliedMatrix[z, j] * (2.0f + this.L1_penality))) + (this.numfeatures * this.L2_penality * this.W[j, z])) / this.M);
                        // Console.WriteLine("Minus");
                        // Console.WriteLine((((-this.multipliedMatrix[z, j] * (2.0f + this.L1_penality))) - (2 * this.L2_penality * this.W[j, z])) / this.M);
                        // Console.WriteLine("No Double");
                        // Console.WriteLine((((-this.multipliedMatrix[z, j] * this.L1_penality)) - (2 * this.L2_penality * this.W[j, z])) / this.M);
                        // Console.WriteLine("Div:");
                        // Console.WriteLine((((-this.multipliedMatrix[z, j]/this.M * (2.0f - this.L1_penality))) + (2 * this.L2_penality * this.W[j, z])) / this.M);
                        // Console.WriteLine("MISC");
                        // Console.WriteLine( (((-1* this.multipliedMatrix[z, j] * - this.L1_penality)) + (2 * this.L2_penality * this.W[j, z])) / this.M);

                        //Console.ReadLine();
                    }
                }
                
            }

            float db = (-(ysum(subtwoarrs(this.Y, Y_pred)))) / this.M;


            // Console.WriteLine("DB");
            // Console.WriteLine(db);
            // Console.ReadLine();
            // Console.ReadLine();

            // Console.WriteLine("Test");
            // this.print2d(this.W);
            // Console.WriteLine();
            // this.print2d(dW);
            // Console.WriteLine("Pre^");
            this.W = subtwo2darrs(this.W, applymul(dW, this.Learning_rate));
            // this.print2d(this.W);
            // Console.ReadLine();
            this.B = this.B - (this.Learning_rate * db);

            this.accelerate.Dispose();
            return this;
        }

        public ElasticRegression update_weightsTILED()
        {
            //Generate a prediction based on inputs
            float[,] Y_pred = this.predictTILED(this.X);
            // Console.WriteLine("Y PRED HERE ");
            // print2d(Y_pred);
            // Console.WriteLine("");
            // Console.WriteLine("");
            // Console.WriteLine("");
            // Console.WriteLine("-----------------");
           
            

            //calculate gradients  
            float[,] dW = new float[this.N, this.Q];
            this.accelerate = this.dev.CreateAccelerator(this.context);

            //calculate gradients  
            this.multipliedMatrix = (MatrixMultiplyTiled(this.accelerate, subtwoarrs(this.Y, Y_pred), this.X));
            for(int j = 0; j < this.N; j++)
            {
                for(int z = 0; z < this.Q; z++){
                    if (this.W[j, z] > 0)
                    {
                        dW[j, z] =  (((-this.multipliedMatrix[z, j] * (2.0f + this.L1_penality))) + (2 * this.L2_penality * this.W[j, z])) / this.M; ;
                    }
                    else
                    {
                        dW[j, z] =  (((-this.multipliedMatrix[z, j] * (2.0f - this.L1_penality))) + (2 * this.L2_penality * this.W[j, z])) / this.M; ; 
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
            this.multipliedMatrix = (matrixmul(subtwoarrs(this.Y, Y_pred), this.X));
            for(int j = 0; j < this.N; j++)
            {
                for(int z = 0; z < this.Q; z++){
                    if (this.W[j, z] > 0)
                    {
                        dW[j, z] =  (((-this.multipliedMatrix[z, j] * (2.0f + this.L1_penality))) + (2 * this.L2_penality * this.W[j, z])) / this.M; ;
                    }
                    else
                    {
                        dW[j, z] =  (((-this.multipliedMatrix[z, j] * (2.0f - this.L1_penality))) + (2 * this.L2_penality * this.W[j, z])) / this.M; ; 
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

            // Console.WriteLine("Prediction");
            // print2d(prediction);
            // Console.WriteLine("this.W");
            // print2d(this.W);
            // Console.ReadLine();
            this.accelerate.Dispose();
            return prediction;
        }
        float[,] predictTILED(float[,] x)
            ///<summary>Predicts output based off of x</summary>
            ///<param name="x">Array of inputs</param>
        { 
            this.accelerate = this.dev.CreateAccelerator(this.context);
            float[,] prediction = applyadd(MatrixMultiplyTiled(this.accelerate, x, this.W), this.B);
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

        const int TILE_SIZE = 2;
        static float[,] MatrixMultiplyTiled(Accelerator accelerator, float[,] a, float[,] b)
        {
            var m = a.GetLength(0);
            var ka = a.GetLength(1);
            var kb = b.GetLength(0);
            var n = b.GetLength(1);

            if (ka != kb)
                throw new ArgumentException($"Cannot multiply {m}x{ka} matrix by {n}x{kb} matrix", nameof(b));

            var kernel = accelerator.LoadStreamKernel<
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(
                MatrixMultiplyTiledKernel);
            var groupSize = new Index2D(TILE_SIZE, TILE_SIZE);
            var numGroups = new Index2D((m + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

            using var aBuffer = accelerator.Allocate2DDenseX<float>(new Index2D(m, ka));
            using var bBuffer = accelerator.Allocate2DDenseX<float>(new Index2D(ka, n));
            using var cBuffer = accelerator.Allocate2DDenseX<float>(new Index2D(m, n));
            aBuffer.CopyFromCPU(a);
            bBuffer.CopyFromCPU(b);

            kernel((numGroups, groupSize), aBuffer, bBuffer, cBuffer);

            // Reads data from the GPU buffer into a new CPU array.
            // Implicitly calls accelerator.DefaultStream.Synchronize() to ensure
            // that the kernel and memory copy are completed first.
            return cBuffer.GetAsArray2D();
        }
        static void MatrixMultiplyTiledKernel(
            ArrayView2D<float, Stride2D.DenseX> aView,
            ArrayView2D<float, Stride2D.DenseX> bView,
            ArrayView2D<float, Stride2D.DenseX> cView)
        {
            var global = Grid.GlobalIndex.XY;
            var x = Group.IdxX;
            var y = Group.IdxY;

            var aTile = SharedMemory.Allocate2D<float, Stride2D.DenseX>(new Index2D(TILE_SIZE, TILE_SIZE), new Stride2D.DenseX(TILE_SIZE));
            var bTile = SharedMemory.Allocate2D<float, Stride2D.DenseX>(new Index2D(TILE_SIZE, TILE_SIZE), new Stride2D.DenseX(TILE_SIZE));
            var sum = 0.0f;

            for (var i = 0; i < aView.IntExtent.X; i += TILE_SIZE)
            {
                if (global.X < aView.IntExtent.X && y + i < aView.IntExtent.Y)
                    aTile[x, y] = aView[global.X, y + i];
                else
                    aTile[x, y] = 0;

                if (x + i < bView.IntExtent.X && global.Y < bView.IntExtent.Y)
                    bTile[x, y] = bView[x + i, global.Y];
                else
                    bTile[x, y] = 0;
                Group.Barrier();

                for (var k = 0; k < TILE_SIZE; k++)
                    sum += aTile[new Index2D(x, k)] * bTile[new Index2D(k, y)];
                Group.Barrier();
            }

            if (global.X < cView.IntExtent.X && global.Y < cView.IntExtent.Y)
                cView[global] = sum;
        }
        void writetoCSV(float[,] array, string path, string inorout){
            StreamWriter file = new StreamWriter(path);
            var iLength = array.GetLength(0);
            var jLength = array.GetLength(1);
            for (int k = 0; k < jLength; k++)
            {
                
                if(k == jLength-1)
                    {
                        file.Write("{1}{0}", k, inorout);
                    }
                    else{
                        file.Write("{1}{0},", k, inorout);
                    }
                
            }
            file.WriteLine();
            for (int j = 0; j < iLength; j++)
            {
                
                for (int i = 0; i < jLength; i++){
                    if(i == jLength-1)
                    {
                        file.Write("{0}", array[j,i]);
                    }
                    else{
                        file.Write("{0},", array[j,i]);
                    }
                    
                }
                file.WriteLine();
                file.Flush();
            }
        }

        void writetoCSVFullClean(float[,] array1, float[,] array2, string path){
            StreamWriter file = new StreamWriter(path);
            var iLength = array1.GetLength(0);
            var jLength = array1.GetLength(1);
            var kLength = array2.GetLength(1);
            for (int k = 0; k < jLength; k++)
            {
                
                if(k == jLength-1)
                    {
                        file.Write("{1}{0},", k, "IN");
                    }
                    else{
                        file.Write("{1}{0},", k, "IN");
                    }
                
            }
            for (int h = 0; h < kLength; h++)
            {
                
                if(h == kLength-1)
                    {
                        file.Write("{1}{0}", h, "OUT");
                    }
                    else{
                        file.Write("{1}{0},", h, "OUT");
                    }
                
            }
            file.WriteLine();
            file.Flush();
            for (int j = 0; j < iLength; j++)
            {
                
                for (int i = 0; i < jLength; i++){
                    if(i == jLength-1)
                    {
                        file.Write("{0},", array1[j,i]);
                    }
                    else{
                        file.Write("{0},", array1[j,i]);
                    }
                    
                }
                for (int z = 0; z < kLength; z++){
                    if(z == kLength-1)
                    {
                        file.Write("{0}", array2[j,z]);
                    }
                    else{
                        file.Write("{0},", array2[j,z]);
                    }
                    
                }
                file.WriteLine();
                file.Flush();
            }
        }
    
        static bool isEqualArrs(float[,] arr1, float[,] arr2){
            int counter = 0;
            if(arr1.GetLength(0) == arr2.GetLength(0) && arr1.GetLength(1) == arr2.GetLength(1)){
                for (int i = 0; i < arr1.GetLength(0); i++)
                {
                    for(int j=0; j < arr1.GetLength(1); j++){
                        if(Math.Abs(arr1[i,j] - arr2[i,j]) > .1){
                            Console.WriteLine(Math.Abs(arr1[i,j] - arr2[i,j]) > .1);
                            Console.Write("Counter");
                            Console.WriteLine(i);
                            Console.WriteLine(j);
                            Console.WriteLine(counter);
                            Console.Write(arr1[i,j]);
                            Console.Write("  ");
                            Console.WriteLine(arr2[i,j]);
                            return false;
                        }
                        else{
                            counter+=1;
                        }
                    //Xactual[i, 1] = ((float)q.NextDouble() * 10) + 2;
                   
                    }
                }
                return true;
            
            }
            else{
                Console.WriteLine("DIFF LENGTHS");
                return false;
            }

        }


        static void Main(string[] args)
        {
            // //learning_rate, iterations, l1_penality, l2_penality 
            ElasticRegression e1 = new ElasticRegression(0.005f, 100, .5f, .05f);
            ElasticRegression e2 = new ElasticRegression(0.005f, 1, .005f, .005f);
            ElasticRegression e3 = new ElasticRegression(0.005f, 10, .5f, .05f);
     
            Random q = new Random();
            //Creates input data
            float[,] Xactual = new float[1000000, 100];
            for (int i = 0; i < Xactual.GetLength(0); i++)
            {
                for(int j=0; j < Xactual.GetLength(1); j++){
                    Xactual[i, j] = ((float)q.NextDouble() * 10) + 2;
                }
                //Xactual[i, 1] = ((float)q.NextDouble() * 10) + 2;
               
            }
            Console.WriteLine("Xactual");
            e1.writetoCSV(Xactual, "BigData1X.csv", "IN");
            //Creates output data
            float[,] Yactual = new float[1000000, 1];
            Console.WriteLine(Yactual.GetLength(0));
            Console.WriteLine(Yactual.GetLength(1));

            for (int i = 0; i < Yactual.GetLength(0); i++)
            {
                for(int j = 0; j < Yactual.GetLength(1); j++){
                    Yactual[i, j] = (((Xactual[i, j]) * 3.2f + 75));
                    //Console.WriteLine(Yactual[i,j]);
                }
                // Yactual[i, 0] = ((Xactual[i, 0]) * 1000 + 2000);
                // Yactual[i, 1] = ((Xactual[i, 0]) * 100 + 100);
                //Yactual[i, 1] = ((Xactual[i, 0]) * 1000 + 2000);
            }
            Console.WriteLine("Yactual");
            e1.writetoCSV(Yactual, "BigData1Y.csv", "OUT");
            // Console.WriteLine(Xactual.GetLength(1));
            // Console.WriteLine(Xactual.GetLength(0));
            // Console.WriteLine(Yactual.GetLength(1));
            // Console.WriteLine(Yactual.GetLength(0));
           

            e1.writetoCSVFullClean(Xactual, Yactual, "BigData1.csv");
            Console.WriteLine("Finshed Building Data");
            
            
        
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


             
            // Stopwatch stopwatch2 = new Stopwatch();


            // stopwatch2.Start();

            Console.WriteLine("Before fit");
            Stopwatch stopw1 = new Stopwatch();
            stopw1.Start();
            e1.fit(Xactual, Yactual);
            stopw1.Stop();

            // Stopwatch stopw2 = new Stopwatch();
            // stopw2.Start();
            // e2.fitTILED(Xactual, Yactual);
            // stopw2.Stop();

            // Stopwatch stopwN = new Stopwatch();
            // stopwN.Start();
            // e3.fitNOGPU(Xactual, Yactual);
            // stopwN.Stop();
            // // stopwatch2.Stop();
            // Console.WriteLine("Without accelerator");
            //


            Stopwatch stopw = new Stopwatch();
            stopw.Start();
            float[,] res = e1.predict(Xactual);
            stopw.Stop();

            // Stopwatch stopw3 = new Stopwatch();
            // stopw3.Start();
            // float[,] res2 = e2.predictTILED(Xactual);
            // stopw3.Stop();

            

            // Stopwatch stopwN1 = new Stopwatch();
            // stopwN1.Start();
            // float[,] resN = e3.predictNOGPU(Xactual);
            // stopwN1.Stop();

            // Console.WriteLine("RES");
            // e2.print2d(res);
            // Console.WriteLine("RES2");
            // e2.print2d(res2);   
            // Console.WriteLine("RESN");
            // e2.print2d(resN);
            // Console.WriteLine("Actual");
            // e2.print2d(Yactual);
            
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();

            //Console.WriteLine(res2[0,1]);
            // Console.WriteLine(isEqualArrs(res,res2));
            // Console.WriteLine(isEqualArrs(res,resN));
            //This prints out the prediction with the actual
            // for (int i = 0; i < res.GetLength(0); i++)
            // {
            //    Console.Write(res[i, 0]);
            //    Console.Write(" | ");
            //    Console.Write(res[i, 1]);
            //    Console.Write(" || ");
            //    Console.Write(resN[i, 0]);  
            //    Console.Write(" | ");
            //    Console.Write(resN[i, 1]);  
            //    Console.Write(" ||| ");
            //    Console.Write(Yactual[i, 0]);
            //    Console.Write(" | ");
            //    Console.Write(Yactual[i,1]);
            //    Console.Write(" ");
            //    //Console.Write(Yactual[i, 1]);

            //    Console.WriteLine();
            // }
            // float res1total = 0;
            // float res2total = 0;
            // float resNtotal = 0;
            // int counter = 0;
            // for(int i = 0; i < res.GetLength(0); i++){
            //     for(int j = 0; j< res.GetLength(1); j++){
            //         res1total += Math.Abs(Yactual[i,j]-res[i,j]);
            //         // res2total += Math.Abs(Yactual[i,j]-res2[i,j]);
            //         // resNtotal += Math.Abs(Yactual[i,j]-resN[i,j]);
            //         counter +=1;

            //     }
            // }

            Console.WriteLine("With GPU:");
            //Console.WriteLine(res1total/counter);
            Console.WriteLine(stopw1.Elapsed);
            Console.WriteLine(stopw.Elapsed);

            // Console.WriteLine("TILED:");
            // Console.WriteLine(res2total/counter);
            // Console.WriteLine(stopw2.Elapsed);
            // Console.WriteLine(stopw3.Elapsed);

            // Console.WriteLine("NO GPU:");
            // Console.WriteLine(resNtotal/counter);
            // Console.WriteLine(stopwN.Elapsed);
            // Console.WriteLine(stopwN1.Elapsed);
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
            //Context context = Context.Create(builder => builder.AllAccelerators());
            // Device dev = context.GetPreferredDevice(preferCPU: false);
            // Accelerator accelerate = dev.CreateAccelerator(context);
            // float[,] arr1 = new float[10, 10];
            // float[,] arr2 = new float[10, 10];

            // for (int i = 0; i < arr1.GetLength(0); i++)
            // {
            //     for(int j = 0; j < arr1.GetLength(0); j++){
            //        arr2[i,j] = 3.2f*i+j;
            //        arr1[i,j] = 1.3f*j-i;
            //     }
            //     // Yactual[i, 0] = ((Xactual[i, 0]) * 1000 + 2000);
            //     // Yactual[i, 1] = ((Xactual[i, 0]) * 100 + 100);
            //     //Yactual[i, 1] = ((Xactual[i, 0]) * 1000 + 2000);
            // }
            // Console.WriteLine("Naive");
            // e1.print2d(e1.matrixmul(arr1, arr2));
            // //matrixmul()
            
            // Console.WriteLine("GPU");
            // e1.print2d(MatrixMultiplyAccelerated(accelerate, arr1, arr2));

            // Console.WriteLine("Tiled");
            // e1.print2d(MatrixMultiplyTiled(accelerate, arr1, arr2));

        }

        
    }
}
