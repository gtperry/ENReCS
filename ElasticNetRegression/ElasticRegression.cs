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
        
        // this.WBuffer = this.accelerate.Allocate2DDenseX<float>(new Index2D(this.N, this.Q));
        //     this.B = 0.0f;
        //     this.X = X;
        //     this.XBuffer = this.accelerate.Allocate2DDenseX<float>(new Index2D( X.GetLength(0),X.GetLength(1)));
        //     XBuffer.CopyFromCPU(X);
        //     this.Y = Y;
        //     this.YPredBuffer = this.accelerate.Allocate2DDenseX<float>(new Index2D( Y.GetLength(0),Y.GetLength(1)));
        //     this.YDiffBuffer = this.accelerate.Allocate2DDenseX<float>(new Index2D( Y.GetLength(0),Y.GetLength(1)));
        //     this.YBuffer = this.accelerate.Allocate2DDenseX<float>(new Index2D( Y.GetLength(0),Y.GetLength(1)));
        //     this.MatMulBuffer = accelerator.Allocate2DDenseX<float>(new Index2D(X.GetLength(0), Y.GetLength(0)));
        //     YBuffer.CopyFromCPU(Y);

        float[,] X;
        MemoryBuffer2D<float, Stride2D.DenseX> XBuffer;
        float[,] Y;
        MemoryBuffer2D<float, Stride2D.DenseX> YBuffer;
        float[,] W;
        MemoryBuffer2D<float, Stride2D.DenseX> WBuffer;
        MemoryBuffer1D<float, Stride1D.Dense> SumBuffer;
        float[,] multipliedMatrix;
        MemoryBuffer2D<float, Stride2D.DenseX> MatMulBuffer;
        MemoryBuffer2D<float, Stride2D.DenseX> PredMatMulBuffer;
        MemoryBuffer2D<float, Stride2D.DenseX> dWBuffer;
        

        //Different Kernals needed for GPU computations
        Action<Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>> subtwoarrkern;
        Action<Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>> subtwoarrkern2;
        MemoryBuffer2D<float, Stride2D.DenseX> YDiffBuffer;
        MemoryBuffer2D<float, Stride2D.DenseX> YPredBuffer;
        Action<Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>> matrixmulkern;
        Action<Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                float, float, float> updatekernel;
        Action<Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                float> matrixaddkern;
        Action<Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                float> matrixmulsinglekern;
        Action<Index1D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView1D<float, Stride1D.Dense>,
                int,int> sumofkern;
        Action<Index1D,
                ArrayView1D<float, Stride1D.Dense>> clearsumkern;




        Device dev;
        Accelerator accelerate;
       
        //Will be used later when optimized for GPU use
        Context context;


        //Constructor
        public ElasticRegression(float learning_rate, int iterations, float l1_penality, float l2_penality, bool fullGPU = false)
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

        public ElasticRegression fitFULLGPU(float[,] X, float[,] Y, bool verbose = true)
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
            this.accelerate = this.dev.CreateAccelerator(this.context);

            this.subtwoarrkern = this.accelerate.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(
                subtwoarrsKernal);
            this.subtwoarrkern2 = this.accelerate.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(
                subtwoarrsKernal2);
            this.matrixmulsinglekern = this.accelerate.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                float>(
                multKernal);
            this.matrixaddkern = this.accelerate.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                float>(
                additionKernal);
            this.matrixmulkern = this.accelerate.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(
                MatrixMultiplyAcceleratedKernel);
            this.updatekernel = this.accelerate.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                float, float, float>(
                updateweightskernal);
            this.sumofkern = this.accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView1D<float, Stride1D.Dense>,
                int,int>(sumofkernal);

            this.clearsumkern = this.accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<float, Stride1D.Dense>>(clearsumbuff);


            //Number of outputs
            this.Q = Y.GetLength(1);
 
            //Initializes variables
            this.W = new float[this.N, this.Q];
            this.WBuffer = this.accelerate.Allocate2DDenseX<float>(new Index2D(this.N, this.Q));
            this.dWBuffer = this.accelerate.Allocate2DDenseX<float>(new Index2D(this.N, this.Q));
            this.B = 0.0f;
            this.X = X;
            this.XBuffer = this.accelerate.Allocate2DDenseX<float>(new Index2D( X.GetLength(0),X.GetLength(1)));
            XBuffer.CopyFromCPU(X);
            this.Y = Y;
            this.YPredBuffer = this.accelerate.Allocate2DDenseX<float>(new Index2D( Y.GetLength(0),Y.GetLength(1)));
            this.YDiffBuffer = this.accelerate.Allocate2DDenseX<float>(new Index2D( Y.GetLength(1),Y.GetLength(0)));
            this.YBuffer = this.accelerate.Allocate2DDenseX<float>(new Index2D(Y.GetLength(0),Y.GetLength(1)));
            Console.WriteLine(X.GetLength(1));
            Console.WriteLine(Y.GetLength(1));
            //Console.ReadLine();
            this.MatMulBuffer = accelerate.Allocate2DDenseX<float>(new Index2D(Y.GetLength(1), X.GetLength(1)));
            this.PredMatMulBuffer = accelerate.Allocate2DDenseX<float>(new Index2D(X.GetLength(0), this.Q));
            this.SumBuffer = accelerate.Allocate1D<float>(1L);
            YBuffer.CopyFromCPU(Y);
            //float[,] Y_pred;
            //float db2 = 0.0f;
            float db = 0.0f;
            //Gradient descent learning
            using(this.YBuffer)
            using(this.YPredBuffer)
            using(this.YDiffBuffer)
            using(this.MatMulBuffer)
            using(this.XBuffer)
            using(this.WBuffer)
            using(this.dWBuffer)
            using(this.SumBuffer)
            using(this.accelerate)
            //using(this.accelerate)
            {
                for(int i = 0; i < this.Iterations; i++)
                {
                    if (verbose)
                    {
                        Console.WriteLine("Iteration {0}/{1}", i, this.Iterations);
                    }

                    //Updates the weights after each iteration
                    //this.update_weightsFULLGPU();

                    //print2d(this.YPredBuffer.GetAsArray2D());
                    //this.InteriorPredict();
                    this.matrixmulkern(this.YPredBuffer.Extent.ToIntIndex(), this.XBuffer, this.WBuffer, this.YPredBuffer);
                    this.matrixaddkern(this.YPredBuffer.Extent.ToIntIndex(), this.YPredBuffer, this.B);
                    //print2d(this.YBuffer.GetAsArray2D());
                    //Console.ReadLine();
                    ///this.accelerate = this.dev.CreateAccelerator(this.context);
                    // Y_pred = this.predict(this.X);
                    // Console.WriteLine("YPredBuffer");
                    // print2d(this.YPredBuffer.GetAsArray2D());
                    // print2d(Y_pred);
                    // Console.ReadLine();
                    // Console.WriteLine();

                    this.subtwoarrkern(this.YBuffer.Extent.ToIntIndex(), this.YBuffer, this.YPredBuffer, this.YDiffBuffer);
                    //print2d(this.YDiffBuffer.GetAsArray2D());
                    //Console.ReadLine();

                    // Console.WriteLine("YDiffBuffer");
                    // print2d(this.YDiffBuffer.GetAsArray2D());
                    // print2d(subtwoarrs(this.Y, Y_pred));
                    // Console.ReadLine();
                    // Console.WriteLine();

                    // this.multipliedMatrix = (matrixmul(subtwoarrs(this.Y, Y_pred), this.X));

                    this.matrixmulkern(this.MatMulBuffer.Extent.ToIntIndex(), this.YDiffBuffer, this.XBuffer, this.MatMulBuffer);
                    
                    // Console.WriteLine("MatMulBuffer");
                    // print2d(this.MatMulBuffer.GetAsArray2D());
                    // print2d(this.multipliedMatrix);
                    // Console.ReadLine();
                    // Console.WriteLine();

                    this.updatekernel(this.WBuffer.Extent.ToIntIndex(), this.WBuffer.View, this.MatMulBuffer.View, this.dWBuffer.View, this.L1_penality, this.L2_penality, this.M);

                    // float[,] dW = new float[this.N, this.Q];
                    // for(int j = 0; j < this.N; j++)
                    // {
                    //     for(int z = 0; z < this.Q; z++){
                    //         if (this.W[j, z] > 0)
                    //         {
                    //             dW[j, z] =  (((-1* this.multipliedMatrix[z, j] * (2.0f + this.L1_penality))) + ( this.L2_penality * this.W[j, z])) / this.M; //(((-this.multipliedMatrix[z, j] * (2.0f + this.L1_penality))) + (2 * this.L2_penality * this.W[j, z])) / this.M; ;
                    
                    //         }
                    //         else
                    //         {
                    //             dW[j, z] =  (((-1 * this.multipliedMatrix[z, j] * (2.0f - this.L1_penality))) + ( this.L2_penality * this.W[j, z])) / this.M; //(((-this.multipliedMatrix[z, j] * (2.0f - this.L1_penality))) + (2 * this.L2_penality * this.W[j, z])) / this.M; ; 

                    //         }
                    //     }
                        
                    // }

                    // Console.WriteLine("DW");
                    // print2d(this.dWBuffer.GetAsArray2D());
                    // print2d(dW);
                    // Console.ReadLine();
                    // Console.WriteLine();

                    this.clearsumkern(this.SumBuffer.Extent.ToIntIndex(), this.SumBuffer.View);
                    //this.sumofkern(this.YDiffBuffer.Extent.ToIntIndex(), this.YDiffBuffer.View, this.SumBuffer.Extent.ToIntIndex(),  this.SumBuffer.View);
                    this.sumofkern(this.SumBuffer.Extent.ToIntIndex(), this.YDiffBuffer.View, this.SumBuffer.View, this.YDiffBuffer.Extent.ToIntIndex().X, this.YDiffBuffer.Extent.ToIntIndex().Y);
                    // Console.WriteLine("SumBuffer");
                    // print1d(this.SumBuffer.GetAsArray1D());
                    // Console.WriteLine(this.SumBuffer.GetAsArray1D()[0]/this.M);
                    //db2 = (-(2.0F * ysum(subtwoarrs(this.Y, Y_pred)))) / this.M;
                    // Console.WriteLine("DB");
                    // Console.WriteLine(ysum(subtwoarrs(this.Y, Y_pred)));
                    // Console.WriteLine(db);
                    // Console.ReadLine();
                    db = (- 2.0F * this.SumBuffer.GetAsArray1D()[0])/this.M;
                    //float db = (-(ysum(subtwoarrs(this.Y, Y_pred)))) / this.M;

                    
                    // Console.WriteLine("DB");
                    // Console.WriteLine(db);
                    // Console.WriteLine(db2);
                    // Console.ReadLine();
                    // Console.ReadLine();

                    // Console.WriteLine("Test");
                    // this.print2d(this.W);
                    // Console.WriteLine();
                    // this.print2d(dW);
                    // Console.WriteLine("Pre^");
                    // Console.WriteLine("WBUffer");
                    // print2d(this.WBuffer.GetAsArray2D());
                    // Console.WriteLine("dWBuffer");
                    // print2d(this.dWBuffer.GetAsArray2D());
                    this.matrixmulsinglekern(this.dWBuffer.Extent.ToIntIndex(), this.dWBuffer, this.Learning_rate);
                    this.subtwoarrkern2(this.WBuffer.Extent.ToIntIndex(), this.WBuffer, this.dWBuffer);

                    //this.W = subtwo2darrs(this.W, applymul(dW, this.Learning_rate));
                    // Console.WriteLine("New W");
                    // print2d(this.WBuffer.GetAsArray2D());
                    // print2d(this.dWBuffer.GetAsArray2D());
                    // print2d(this.W);
                    // Console.ReadLine();

                    // Console.WriteLine("UpdatedW Bufffer");
                    // print2d(this.dWBuffer.GetAsArray2D());
                    //this.W = subtwo2darrs(this.W, applymul(dW, this.Learning_rate));
                    // this.print2d(this.W);
                    // Console.ReadLine();
                    this.B = this.B - (this.Learning_rate * db);

                }
                this.W = this.WBuffer.GetAsArray2D();
            }
            // print2d(this.W);
            // Console.ReadLine();
            
            this.accelerate.Dispose();
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
            
                    }
                    else
                    {
                        dW[j, z] =  (((-1 * this.multipliedMatrix[z, j] * (2.0f - this.L1_penality))) + ( this.L2_penality * this.W[j, z])) / this.M; //(((-this.multipliedMatrix[z, j] * (2.0f - this.L1_penality))) + (2 * this.L2_penality * this.W[j, z])) / this.M; ; 

                    }
                }
                
            }

            float db = (-(2.0F * ysum(subtwoarrs(this.Y, Y_pred)))) / this.M;


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



        public ElasticRegression update_weightsFULLGPU()
        {
            //Generate a prediction based on inputs
            //float[,] Y_pred = this.predict(this.X);
            Console.WriteLine("YPREDBUFFer");
            using(this.YBuffer)
            using(this.YPredBuffer)
            using(this.YDiffBuffer)
            using(this.MatMulBuffer)
            using(this.XBuffer)
            using(this.WBuffer)
            using(this.dWBuffer)
            using(this.SumBuffer)
            using(this.accelerate)
            //using(this.accelerate)
            {
                print2d(this.YPredBuffer.GetAsArray2D());
                //this.InteriorPredict();
                this.matrixmulkern(this.YPredBuffer.Extent.ToIntIndex(), this.XBuffer, this.WBuffer, this.YPredBuffer);
                this.matrixaddkern(this.YPredBuffer.Extent.ToIntIndex(), this.YPredBuffer, this.B);
                print2d(this.YBuffer.GetAsArray2D());
                Console.ReadLine();
                //this.accelerate = this.dev.CreateAccelerator(this.context);

                Console.WriteLine("YDiffBuffer");
                print2d(this.YDiffBuffer.GetAsArray2D());
                

                this.subtwoarrkern(this.YBuffer.Extent.ToIntIndex(), this.YBuffer, this.YPredBuffer, this.YDiffBuffer);
                print2d(this.YDiffBuffer.GetAsArray2D());
                Console.ReadLine();
                

                this.matrixmulkern(this.MatMulBuffer.Extent.ToIntIndex(), this.YDiffBuffer, this.XBuffer, this.MatMulBuffer);
                //Multiplied matrix has # of Youtputs rows and # of X features columns
                //this.multipliedMatrix = (MatrixMultiplyAccelerated(this.accelerate, subtwoarrs(this.Y, Y_pred), this.X));
            
                //float[,] dW = new float[this.N, this.Q];
                ///Can make this a kernal
                // for(int j = 0; j < this.N; j++)
                // {
                //     for(int z = 0; z < this.Q; z++){
                //         if (this.W[j, z] > 0)
                //         {

                //             dW[j, z] =  (((-1* this.MatMulBuffer[new Index2D(z,j)] * (2.0f + this.L1_penality))) + ( this.L2_penality * this.WBuffer[new Index2D(j,z)])) / this.M; //(((-this.multipliedMatrix[z, j] * (2.0f + this.L1_penality))) + (2 * this.L2_penality * this.W[j, z])) / this.M; ;

                //         }
                //         else
                //         {
                //             dW[j, z] =  (((-1 * this.multipliedMatrix[z, j] * (2.0f - this.L1_penality))) + ( this.L2_penality * this.W[j, z])) / this.M; //(((-this.multipliedMatrix[z, j] * (2.0f - this.L1_penality))) + (2 * this.L2_penality * this.W[j, z])) / this.M; ; 

                //         }
                //     }
                    
                // }
                
                this.updatekernel(this.WBuffer.Extent.ToIntIndex(), this.WBuffer.View, this.MatMulBuffer.View, this.dWBuffer.View, this.L1_penality, this.L2_penality, this.M);
                this.clearsumkern(this.SumBuffer.Extent.ToIntIndex(), this.SumBuffer.View);
                //this.sumofkern(this.YDiffBuffer.Extent.ToIntIndex(), this.YDiffBuffer.View,this.SumBuffer.Extent.ToIntIndex(), this.SumBuffer.View);
                this.sumofkern(this.SumBuffer.Extent.ToIntIndex(), this.YDiffBuffer.View, this.SumBuffer.View, this.YDiffBuffer.Extent.ToIntIndex().X, this.YDiffBuffer.Extent.ToIntIndex().Y);
                float db = (- 2.0F * this.SumBuffer.GetAsArray1D()[0])/this.M;
                //float db = (-(ysum(subtwoarrs(this.Y, Y_pred)))) / this.M;


                // Console.WriteLine("DB");
                // Console.WriteLine(db);
                // Console.ReadLine();
                // Console.ReadLine();

                // Console.WriteLine("Test");
                // this.print2d(this.W);
                // Console.WriteLine();
                // this.print2d(dW);
                // Console.WriteLine("Pre^");
                this.subtwoarrkern(this.WBuffer.Extent.ToIntIndex(), this.WBuffer, this.dWBuffer, this.WBuffer);
                //this.W = subtwo2darrs(this.W, applymul(dW, this.Learning_rate));
                // this.print2d(this.W);
                // Console.ReadLine();
                this.B = this.B - (this.Learning_rate * db);

            }
            return this;
        }

        static void clearsumbuff(Index1D index, ArrayView1D<float, Stride1D.Dense> sum){
            sum[index] = 0.0f;
        }
        static void sumofkernal(Index1D index, ArrayView2D<float, Stride2D.DenseX> aView,ArrayView1D<float, Stride1D.Dense> sum, int dim1, int dim2){
            for(int i = 0; i < dim1; i++){
               for(int j = 0; j < dim2; j++){
                    sum[index] += aView[new Index2D(i,j)];
                } 
            }
            

            
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



        static float[,] kernalTester(Accelerator accelerator, float[,] a, float[,] b){
            var m = a.GetLength(0);
            var ka = a.GetLength(1);
            var kb = b.GetLength(0);
            var n = b.GetLength(1);

            var kernal =  accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>, 
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(
                subtwoarrsKernal);

            using var aBuffer = accelerator.Allocate2DDenseX<float>(new Index2D(m, ka));
            using var bBuffer = accelerator.Allocate2DDenseX<float>(new Index2D(m, ka));
            using var cBuffer = accelerator.Allocate2DDenseX<float>(new Index2D(m, ka));

            aBuffer.CopyFromCPU(a);
            bBuffer.CopyFromCPU(b);

            kernal(aBuffer.Extent.ToIntIndex(),aBuffer.View, bBuffer.View, cBuffer.View);

            return cBuffer.GetAsArray2D();
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

        static void updateweightskernal(Index2D index,  
            ArrayView2D<float, Stride2D.DenseX> WView,
            ArrayView2D<float, Stride2D.DenseX> MMView,
            ArrayView2D<float, Stride2D.DenseX> DwView,
            float L1,
            float L2,
            float M){

            if (WView[index] > 0)
            {

                DwView[index] =  (((-1* MMView[new Index2D(index.Y,index.X)] * (2.0f + L1))) + (L2 * WView[index])) / M; //(((-this.multipliedMatrix[z, j] * (2.0f + this.L1_penality))) + (2 * this.L2_penality * this.W[j, z])) / this.M; ;

            }
            else
            {
                DwView[index] =  (((-1 * MMView[new Index2D(index.Y,index.X)] * (2.0f - L1))) + (L2 * WView[index])) / M; //(((-this.multipliedMatrix[z, j] * (2.0f - this.L1_penality))) + (2 * this.L2_penality * this.W[j, z])) / this.M; ; 

            }

        }
        static void additionKernal(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> aView,
            float addvalue
            ){
            // var  xindex = index.X;
            // //var sum = 0.0f;
            // for (var i = 0; i < aView.IntExtent.Y; i++){
            //      aView[new Index2D(xindex,i)] = aView[new Index2D(xindex,i)] + addvalue;
            // }
            //bView[index] = sum;

            aView[index] += addvalue;
        }
        static void multKernal(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> aView,
            float multvalue
            ){
            aView[index] = aView[index] * multvalue;

        }

        static void subKernal(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> aView,
            float subvalue
            ){

            aView[index] = aView[index] - subvalue;
           

        }
        static void divKernal(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> aView,
            float divvalue
            ){
            aView[index] = aView[index]/divvalue;
            

        }

        static void subtwoarrsKernal(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> aView,
            ArrayView2D<float, Stride2D.DenseX> bView,
            ArrayView2D<float, Stride2D.DenseX> cView){
            // var xindex = index.X;
            
            // for (var i= 0; i< aView.IntExtent.Y; i++){
                
            //     cView[new Index2D(i, xindex)] = aView[new Index2D(xindex, i)] - bView[new Index2D(xindex, i)];
            // }
            
            //cView[new Index2D(index.Y, index.X)] = aView[index] - bView[index];
            cView[new Index2D(index.Y, index.X)] = aView[index] - bView[index];
            
        }
        static void subtwoarrsKernal2(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> aView,
            ArrayView2D<float, Stride2D.DenseX> bView){
            // var xindex = index.X;
            
            // for (var i= 0; i< aView.IntExtent.Y; i++){
                
            //     cView[new Index2D(i, xindex)] = aView[new Index2D(xindex, i)] - bView[new Index2D(xindex, i)];
            // }
            
            //cView[new Index2D(index.Y, index.X)] = aView[index] - bView[index];
            aView[index] = aView[index] - bView[index];
            
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
        static void MatMulKernal(
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
        /*
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
        */
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
        float[,] predictFULLGPU(float[,] x)
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
        void InteriorPredict()
            ///<summary>Predicts output based off of x</summary>
            ///<param name="x">Array of inputs</param>
        { 

            // this.matrixaddkern = this.accelerate.LoadAutoGroupedStreamKernel<
            //     Index2D,
            //     ArrayView2D<float, Stride2D.DenseX>,
            //     float>(
            //     additionKernal);
            // this.matrixmulkern = this.accelerate.LoadAutoGroupedStreamKernel<
            //     Index2D,
            //     ArrayView2D<float, Stride2D.DenseX>,
            //     ArrayView2D<float, Stride2D.DenseX>,
            //     ArrayView2D<float, Stride2D.DenseX>>(
            //     MatrixMultiplyAcceleratedKernel);
            Console.WriteLine("Here");
            using(this.YPredBuffer)
            using(this.YBuffer)
            using(this.WBuffer){
                this.matrixmulkern(this.YPredBuffer.Extent.ToIntIndex(), this.XBuffer, this.WBuffer, this.YPredBuffer);
                this.matrixaddkern(this.YPredBuffer.Extent.ToIntIndex(), this.YPredBuffer, this.B);
            }
            Console.WriteLine("oUT");
            //float[,] prediction = applyadd(MatrixMultiplyAccelerated(this.accelerate, x, this.W), this.B);

            // Console.WriteLine("Prediction");
            // print2d(prediction);
            // Console.WriteLine("this.W");
            // print2d(this.W);
            // Console.ReadLine();
            //this.accelerate.Dispose();
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
                for (int j = 0; j < y.GetLength(0); j++){
                    total = total + y[j, i];
                }    
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
        float sumof(float[,] arr, int row){
            float sum = 0.0f;
            for(int i = 0; i < arr.GetLength(1); i++){
                sum += arr[row, i];
            }
            return sum;
        }

        static void Main(string[] args)
        {
            // //learning_rate, iterations, l1_penality, l2_penality 
            ElasticRegression e1 = new ElasticRegression(0.005f, 100, 1.5f,  .5f);
            ElasticRegression e2 = new ElasticRegression(0.005f, 100, 1.5f, .5f);
            ElasticRegression e3 = new ElasticRegression(0.005f, 10, .5f, .5f);
        

            // float[,] kernaltesta = new float[10,10];
            // float[,] kernaltestb = new float[10,10];
            // float county = 0.0f;
            // for (int i = 0; i < kernaltesta.GetLength(0); i++)
            // {
            //     for(int j=0; j < kernaltesta.GetLength(1); j++){
            //         kernaltesta[i, j] = 5.4f + county;
            //         kernaltestb[i, j] = 10.3f + county;
            //         county +=1;

            //     }
            //     //Xactual[i, 1] = ((float)q.NextDouble() * 10) + 2;
               
            // }
            // Context context = Context.Create(builder => builder.AllAccelerators());
            // Device dev = context.GetPreferredDevice(preferCPU: false);
            // Accelerator aa = dev.CreateAccelerator(context);
            // e1.print2d(kernalTester(aa, kernaltesta, kernaltestb));
            // Console.ReadLine();
            // aa.Dispose();

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
            e1.writetoCSV(Xactual, "BigData2X.csv", "IN");
            //Creates output data
            float[,] Yactual = new float[1000000, 1];
            Console.WriteLine(Yactual.GetLength(0));
            Console.WriteLine(Yactual.GetLength(1));

            for (int i = 0; i < Yactual.GetLength(0); i++)
            {
                for(int j = 0; j < Yactual.GetLength(1); j++){
                    Yactual[i, j] = ((e1.sumof(Xactual, i) * 3.2f + 75));
                    //Console.WriteLine(Yactual[i,j]);
                }
                // Yactual[i, 0] = ((Xactual[i, 0]) * 1000 + 2000);
                // Yactual[i, 1] = ((Xactual[i, 0]) * 100 + 100);
                //Yactual[i, 1] = ((Xactual[i, 0]) * 1000 + 2000);
            }
            Console.WriteLine("Yactual");
            e1.writetoCSV(Yactual, "BigData2Y.csv", "OUT");
            // Console.WriteLine(Xactual.GetLength(1));
            // Console.WriteLine(Xactual.GetLength(0));
            // Console.WriteLine(Yactual.GetLength(1));
            // Console.WriteLine(Yactual.GetLength(0));
           

            e1.writetoCSVFullClean(Xactual, Yactual, "BigData2.csv");
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
            Stopwatch stopw2 = new Stopwatch();
            stopw2.Start();
            e2.fitFULLGPU(Xactual, Yactual);
            stopw2.Stop();


            Console.WriteLine("Before fit");
            Stopwatch stopw1 = new Stopwatch();
            stopw1.Start();
            e1.fit(Xactual, Yactual);
            stopw1.Stop();

            
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

            Stopwatch stopw3 = new Stopwatch();
            stopw3.Start();
            float[,] res2 = e2.predict(Xactual);
            stopw3.Stop();

            

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
           float res1total = 0;
           float res2total = 0;
            //float resNtotal = 0;
            int counter = 0;
            for(int i = 0; i < res.GetLength(0); i++){
                for(int j = 0; j< res.GetLength(1); j++){
                    Console.Write(Yactual[i,j]);
                    Console.Write(" | ");
                    Console.WriteLine(res2[i,j]);
                    res1total += Math.Abs(Yactual[i,j]-res[i,j]);
                    res2total += Math.Abs(Yactual[i,j]-res2[i,j]);
                    // resNtotal += Math.Abs(Yactual[i,j]-resN[i,j]);
                    counter +=1;

                }
            }

            Console.WriteLine("With GPU:");
            Console.WriteLine(res1total/counter);
            Console.WriteLine(stopw1.Elapsed);
            Console.WriteLine(stopw.Elapsed);

            Console.WriteLine("FULL GPU:");
            Console.WriteLine(res2total/counter);
            Console.WriteLine(stopw2.Elapsed);
            Console.WriteLine(stopw3.Elapsed);

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
