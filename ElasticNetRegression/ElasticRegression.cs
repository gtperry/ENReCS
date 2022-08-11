
using System;
using System.IO;
using System.Threading.Tasks;
using System.Diagnostics;

using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;

using System.Globalization;
using CsvHelper;
using CsvHelper.Configuration;

namespace ElasticNetRegression
{
    public class ElasticNet
    {
        Device dev;
        Accelerator accelerate;

        //Will be used later when optimized for GPU use
        Context context;

        int N;
        int P;
        public double B;
        public double[] W;
        const int COLLIMIT = 10000;
        const int ROWLIMIT = 10000;

        MemoryBuffer2D<double, Stride2D.DenseX> XBuffer;
        MemoryBuffer2D<double, Stride2D.DenseX> X2Buffer;
        MemoryBuffer2D<double, Stride2D.DenseX> X2TransposeBuffer;
        MemoryBuffer1D<double, Stride1D.Dense> ColMeansBuffer;
        MemoryBuffer1D<double, Stride1D.Dense> ColSTDBuffer;
        MemoryBuffer1D<double, Stride1D.Dense> YBuffer; // y
        MemoryBuffer1D<double, Stride1D.Dense> YMeanBuffer; // ym
        MemoryBuffer1D<double, Stride1D.Dense> YNormBuffer; // Y

        MemoryBuffer1D<double, Stride1D.Dense> WBuffer;
        MemoryBuffer1D<double, Stride1D.Dense> UBuffer;
        MemoryBuffer1D<double, Stride1D.Dense> ZBuffer;
        MemoryBuffer2D<double, Stride2D.DenseX> FBuffer;
        MemoryBuffer1D<double, Stride1D.Dense> NewWBuffer;
        MemoryBuffer1D<double, Stride1D.Dense> NewUBuffer;
        MemoryBuffer1D<double, Stride1D.Dense> NewZBuffer;
        MemoryBuffer2D<double, Stride2D.DenseX> NewFBuffer;

        MemoryBuffer1D<double, Stride1D.Dense> MaxValBuffer;

        MemoryBuffer1D<double, Stride1D.Dense> DXBuffer;
        MemoryBuffer1D<double, Stride1D.Dense> DUBuffer;
        MemoryBuffer1D<double, Stride1D.Dense> DXUBuffer;
        MemoryBuffer1D<double, Stride1D.Dense> GradBuffer;

        MemoryBuffer1D<double, Stride1D.Dense> DiagxtxBuffer;

        MemoryBuffer1D<double, Stride1D.Dense> NuBuffer;
        MemoryBuffer1D<double, Stride1D.Dense> XNuBuffer;

        MemoryBuffer1D<double, Stride1D.Dense> Q1Buffer;
        MemoryBuffer1D<double, Stride1D.Dense> Q2Buffer;
        MemoryBuffer1D<double, Stride1D.Dense> D1Buffer;
        MemoryBuffer1D<double, Stride1D.Dense> D2Buffer;

        MemoryBuffer2D<double, Stride2D.DenseX> GradphiBuffer;
        MemoryBuffer1D<double, Stride1D.Dense> PrbBuffer;
        MemoryBuffer1D<double, Stride1D.Dense> PrsBuffer;

        MemoryBuffer1D<double, Stride1D.Dense> Norm1Buffer;
        MemoryBuffer1D<double, Stride1D.Dense> GradNormBuffer;
        MemoryBuffer1D<double, Stride1D.Dense> DotProdBufferZxZ;
        MemoryBuffer1D<double, Stride1D.Dense> DotProdBufferNUxNU;
        MemoryBuffer1D<double, Stride1D.Dense> DotProdBufferNUxY;
        MemoryBuffer1D<double, Stride1D.Dense> DotProdBufferGradxDXU;
        MemoryBuffer1D<double, Stride1D.Dense> DotProdBufferWxCenter;

        MemoryBuffer1D<double, Stride1D.Dense> PCGBufferP;
        MemoryBuffer1D<double, Stride1D.Dense> PCGBufferPP;
        MemoryBuffer1D<double, Stride1D.Dense> PCGBufferR;
        MemoryBuffer1D<double, Stride1D.Dense> PCGBufferRR;
        MemoryBuffer1D<double, Stride1D.Dense> PCGBufferZ;
        MemoryBuffer1D<double, Stride1D.Dense> PCGBufferZZ;

        MemoryBuffer1D<double, Stride1D.Dense> AXBuffer;
        MemoryBuffer1D<double, Stride1D.Dense> ATAXBuffer;
        MemoryBuffer2D<double, Stride2D.DenseX> ATABuffer;

        MemoryBuffer1D<double, Stride1D.Dense> BNormBuffer;
        MemoryBuffer1D<double, Stride1D.Dense> BKNumBuffer;
        MemoryBuffer1D<double, Stride1D.Dense> BKBuffer;
        MemoryBuffer1D<double, Stride1D.Dense> BKDenBuffer;

        MemoryBuffer1D<double, Stride1D.Dense> RNormBuffer;
        MemoryBuffer1D<double, Stride1D.Dense> ErrBuffer;

        MemoryBuffer1D<double, Stride1D.Dense> AKBuffer;
        MemoryBuffer1D<double, Stride1D.Dense> AKDenBuffer;



        MemoryBuffer1D<double, Stride1D.Dense> PhiBuffer;
        MemoryBuffer1D<double, Stride1D.Dense> NewPhiBuffer;
        MemoryBuffer1D<double, Stride1D.Dense> USumBuffer;
        MemoryBuffer1D<double, Stride1D.Dense> FSumlgBuffer;
        MemoryBuffer1D<double, Stride1D.Dense> FMaxBuffer;
        MemoryBuffer1D<double, Stride1D.Dense> TestBuffer;
        MemoryBuffer1D<int, Stride1D.Dense> CompPhiBuffer;

        
        

        public ElasticNet()
        {
            this.context = Context.Create(builder => builder.AllAccelerators().EnableAlgorithms());

            this.dev = this.context.GetPreferredDevice(preferCPU: false);
        }
        public ElasticNet fit(double[,] X, double[] Y, double lambda1, double lambda2, double tol, int maxIter = 1000, bool verbose = true)
        {
            // Console.WriteLine("Here");
            Stopwatch getasarraytimes = new Stopwatch();
            Stopwatch timeonfunction = new Stopwatch();
            //double lambda2 = 0.5f;
            timeonfunction.Start();
            this.P = X.GetLength(1);
            this.N = X.GetLength(0);
            int X2length = this.N + this.P;

            
            double ymeansum = 0.0;
            double[] Y2 = new double[this.N + this.P];
            Array.Fill(Y2, 0.0);
            //Fills the y2 array
            for (int i = 0; i < this.N; i++)
            {
                
                Y2[i] = Y[i];
                ymeansum += Y[i];

            }
            

            double c = 1 / (Math.Sqrt(1 + lambda2));

            double padding = c * Math.Sqrt(lambda2);

            int n = X.GetLength(0);
            int p = X.GetLength(1);
            
            //Initialize the buffers that are used later
            this.accelerate = this.dev.CreateAccelerator(this.context);
            this.XBuffer = this.accelerate.Allocate2DDenseX<double>(new Index2D(n, p));
            this.ColMeansBuffer = this.accelerate.Allocate1D<double>((long)X.GetLength(1));
            this.ColSTDBuffer = this.accelerate.Allocate1D<double>((long)X.GetLength(1));
            XBuffer.CopyFromCPU(X);
            this.YBuffer = this.accelerate.Allocate1D<double>(new Index1D(Y2.GetLength(0)));
            this.YNormBuffer = this.accelerate.Allocate1D<double>(new Index1D(Y2.GetLength(0)));
            this.YMeanBuffer = this.accelerate.Allocate1D<double>(new Index1D(1));
            YBuffer.CopyFromCPU(Y2);
            this.X2Buffer = this.accelerate.Allocate2DDenseX<double>(new Index2D(X2length, X.GetLength(1)));
            this.X2TransposeBuffer = this.accelerate.Allocate2DDenseX<double>(new Index2D(X.GetLength(1), X2length));
            this.AXBuffer = this.accelerate.Allocate1D<double>(new Index1D(X2length));
            this.ATAXBuffer = this.accelerate.Allocate1D<double>(new Index1D(X.GetLength(1)));
            this.ATABuffer = this.accelerate.Allocate2DDenseX<double>(new Index2D(X.GetLength(1), X.GetLength(1)));
            this.MaxValBuffer = this.accelerate.Allocate1D<double>(new Index1D(1));
            this.WBuffer = this.accelerate.Allocate1D<double>(new Index1D(X.GetLength(1)));
            this.UBuffer = this.accelerate.Allocate1D<double>(new Index1D(X.GetLength(1)));
            this.ZBuffer = this.accelerate.Allocate1D<double>(new Index1D(Y2.GetLength(0)));
            this.FBuffer = this.accelerate.Allocate2DDenseX<double>(new Index2D(2, X.GetLength(1)));
            this.NewWBuffer = this.accelerate.Allocate1D<double>(new Index1D(X.GetLength(1)));
            this.NewUBuffer = this.accelerate.Allocate1D<double>(new Index1D(X.GetLength(1)));
            this.NewZBuffer = this.accelerate.Allocate1D<double>(new Index1D(Y2.GetLength(0)));
            this.NewFBuffer = this.accelerate.Allocate2DDenseX<double>(new Index2D(2, X.GetLength(1)));
            this.DXBuffer = this.accelerate.Allocate1D<double>(new Index1D(X.GetLength(1)));
            this.DUBuffer = this.accelerate.Allocate1D<double>(new Index1D(X.GetLength(1)));
            this.DXUBuffer = this.accelerate.Allocate1D<double>(new Index1D(X.GetLength(1) * 2));
            this.GradBuffer = this.accelerate.Allocate1D<double>(new Index1D(X.GetLength(1) * 2));
            this.DiagxtxBuffer = this.accelerate.Allocate1D<double>(new Index1D(X.GetLength(1)));
            this.NuBuffer = this.accelerate.Allocate1D<double>(new Index1D(X2length));
            this.XNuBuffer = this.accelerate.Allocate1D<double>(new Index1D(X.GetLength(1)));
            var TESTXNuBuffer = this.accelerate.Allocate1D<double>(new Index1D(X.GetLength(1)));
            this.Q1Buffer = this.accelerate.Allocate1D<double>(new Index1D(X.GetLength(1)));
            this.Q2Buffer = this.accelerate.Allocate1D<double>(new Index1D(X.GetLength(1)));
            this.D1Buffer = this.accelerate.Allocate1D<double>(new Index1D(X.GetLength(1)));
            this.D2Buffer = this.accelerate.Allocate1D<double>(new Index1D(X.GetLength(1)));
            this.GradphiBuffer = this.accelerate.Allocate2DDenseX<double>(new Index2D(2, X.GetLength(1)));
            this.PrbBuffer = this.accelerate.Allocate1D<double>(new Index1D(X.GetLength(1)));
            this.PrsBuffer = this.accelerate.Allocate1D<double>(new Index1D(X.GetLength(1)));
            this.Norm1Buffer = this.accelerate.Allocate1D<double>(new Index1D(1));
            this.GradNormBuffer = this.accelerate.Allocate1D<double>(new Index1D(1));
            this.DotProdBufferZxZ = this.accelerate.Allocate1D<double>(new Index1D(1));
            this.DotProdBufferNUxNU = this.accelerate.Allocate1D<double>(new Index1D(1));
            this.DotProdBufferNUxY = this.accelerate.Allocate1D<double>(new Index1D(1));
            this.DotProdBufferGradxDXU = this.accelerate.Allocate1D<double>(new Index1D(1));
            this.DotProdBufferWxCenter = this.accelerate.Allocate1D<double>(new Index1D(1));
            this.PCGBufferP = this.accelerate.Allocate1D<double>(new Index1D(X.GetLength(1) * 2));
            this.PCGBufferPP = this.accelerate.Allocate1D<double>(new Index1D(X.GetLength(1) * 2));
            this.PCGBufferR = this.accelerate.Allocate1D<double>(new Index1D(X.GetLength(1) * 2));
            this.PCGBufferRR = this.accelerate.Allocate1D<double>(new Index1D(X.GetLength(1) * 2));
            this.PCGBufferZ = this.accelerate.Allocate1D<double>(new Index1D(X.GetLength(1) * 2));
            this.PCGBufferZZ = this.accelerate.Allocate1D<double>(new Index1D(X.GetLength(1) * 2));
            this.BNormBuffer = this.accelerate.Allocate1D<double>(new Index1D(1));
            this.BKNumBuffer = this.accelerate.Allocate1D<double>(new Index1D(1));
            this.BKBuffer = this.accelerate.Allocate1D<double>(new Index1D(1));
            this.BKDenBuffer = this.accelerate.Allocate1D<double>(new Index1D(1));
            this.RNormBuffer = this.accelerate.Allocate1D<double>(new Index1D(1));
            this.ErrBuffer = this.accelerate.Allocate1D<double>(new Index1D(1));
            this.AKBuffer = this.accelerate.Allocate1D<double>(new Index1D(1));
            this.AKDenBuffer = this.accelerate.Allocate1D<double>(new Index1D(1));
            this.PhiBuffer = this.accelerate.Allocate1D<double>(new Index1D(1));
            this.NewPhiBuffer = this.accelerate.Allocate1D<double>(new Index1D(1));
            this.USumBuffer = this.accelerate.Allocate1D<double>(new Index1D(45));
            this.FSumlgBuffer = this.accelerate.Allocate1D<double>(new Index1D(1));
            this.FMaxBuffer = this.accelerate.Allocate1D<double>(new Index1D(1));
            this.TestBuffer = this.accelerate.Allocate1D<double>(new Index1D(1));
            this.CompPhiBuffer = this.accelerate.Allocate1D<int>(new Index1D(1));


            //Initializes all of the kernals for use later in the algo
            var columnMeansKern = this.accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView1D<double, Stride1D.Dense>,
                int>(
                columnMeansKernal);
   
            var columnMeans1DKern = this.accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                int>(
                columnMeans1DKernal);


            var columnSTDevKern = this.accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                int>(
                columnSTDevKernal);

            
            var x2fillkern = this.accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                double, double, int, int>(
                fillX2Kernal);

            var subByColumnsKern = this.accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>>(
                subByColumnsKernal);

            var setBuffToValueKern = this.accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                double>(
                setBuffToValueKernal);

            var FFillKern = this.accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView2D<double, Stride2D.DenseX>>(
                FFillKernal);

            var matrixmulkern = this.accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>>(
                MatrixMultiplyAcceleratedKernel);

            var matrixmul2dkern = this.accelerate.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView2D<double, Stride2D.DenseX>>(
                MatrixMultiply2DKernel);

            var matrixmulGradphikern = this.accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView2D<double, Stride2D.DenseX>>(
                MatrixMultiplyGradphiKernel);

            var InitializeNuKern = this.accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>>(
                InitializeNuKernal);

            var GetMaxValKern = this.accelerate.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                int>(GetMaxValKernal);

            var DualityGapKern = this.accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                double,
                int>(DualityGapKernal);

            var PobjKern = this.accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                double>(PobjKernal);

            var DobjKern = this.accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                double>(DobjKernal);


            var NormKern = this.accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                int>(NormKernal);

            var NewtonStepKern = this.accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                double>(NewtonStepKernal);

            var GradPhiKern = this.accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                double, double, int>(GradPhiKernal);

            var fillInverseMatrixKern = this.accelerate.LoadAutoGroupedStreamKernel<Index2D,
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView2D<double, Stride2D.DenseX>>(fillInverseMatrixKernal);

            var PCGmvKern = this.accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                int
                >(PCGmvKernal);

            var PCGasolveKern = this.accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                int>(PCGasolveKernal);

            var RandRRFillKern = this.accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>>(RandRRFillKernal);

            var FillPsKern = this.accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>>(FillPsKernal);


            var FirstIterationFillPsKern = this.accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>>(FirstIterationFillPsKernal);


            var CopyBufferKern2D = this.accelerate.LoadAutoGroupedStreamKernel<Index2D,
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView2D<double, Stride2D.DenseX>>(CopyBufferKernal2D);


            var CopyBufferKern1D = this.accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>>(CopyBufferKernal1D);


            var SubFromRsKern = this.accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>>(SubFromRsKernal);


            var PreconditionerVectorKern = this.accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>>(PreconditionerVectorKernal);

            

            var calcErrorKern = this.accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>>(calcErrorKernal);


            var splitDXUKern = this.accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>, int>(splitDXUKernal);


            var setnewBuffersKern = this.accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView1D<double, Stride1D.Dense>,
                double>(setnewBuffersKernal);

            var CalcphiKern = this.accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                double, double>(CalcphiKernal);

            var SubYFromZKern = this.accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>>(SubYFromZKernal);

            var WFinaleKern = this.accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                double>(WFinaleKernal);

            var nphicompkern= this.accelerate.LoadAutoGroupedStreamKernel<Index1D , 
                ArrayView1D<double, Stride1D.Dense> ,
                ArrayView1D<double, Stride1D.Dense> ,
                ArrayView1D<double, Stride1D.Dense> ,
                ArrayView1D<int, Stride1D.Dense> ,
                double ,
                double>(nphicompkernal);

            var TripleDotKern= this.accelerate.LoadStreamKernel<
                ArrayView<double>,
                ArrayView<double>,
                ArrayView<double>,
                ArrayView<double>,
                ArrayView<double>,
                ArrayView<double>,
                ArrayView<double>,
                int >(TripleDotKernal);

            var SingleDotKern= this.accelerate.LoadStreamKernel<
                ArrayView<double>,
                ArrayView<double>,
                ArrayView<double>,
                int >(SingleDotKernal);

            var SingleSelfDotKern= this.accelerate.LoadStreamKernel<
                ArrayView<double>,
                ArrayView<double>,
                int>(SingleSelfDotKernal);

            var sharedMemArrKern = this.accelerate.LoadStreamKernel<
                ArrayView<double>, 
                ArrayView<double>>(SharedMemoryArrayKernel);

            var SharedMemoryVariableKern= this.accelerate.LoadStreamKernel<
                ArrayView<double>,          
                ArrayView<double>,int,int>(SharedMemoryVariableKerneL); 

            var SMSumOfKern = this.accelerate.LoadStreamKernel<
                ArrayView<double>,          
                ArrayView<double>,int>(SMSumOfKernal); 

            var SMNormKern = this.accelerate.LoadStreamKernel<
                ArrayView<double>,          
                ArrayView<double>,int>(SMNormKernal); 

            var SMSumLogNegKern= this.accelerate.LoadStreamKernel<
                ArrayView2D<double, Stride2D.DenseX>,          
                ArrayView<double>,
                int>(SMSumLogNegKernal);

            var sqrtkern= this.accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<double, Stride1D.Dense>>(sqrtkernal);

            var SMMatMul2DKern  = this.accelerate.LoadAutoGroupedStreamKernel<Index3D,
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView2D<double, Stride2D.DenseX>>(SMMatMul2DKernal);

            var SMMatMul1DKern  = this.accelerate.LoadAutoGroupedStreamKernel<Index2D,
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>>(SMMatMul1DKernal);

            var SMMatMul1DKernTooLarge  = this.accelerate.LoadAutoGroupedStreamKernel<Index2D,
                ArrayView2D<double, Stride2D.DenseX>,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>>(SMMatMul1DKernalTooLarge);
                
            
            timeonfunction.Stop();


            double MU = 2.0;
            double ALPHA = 0.01;
            double BETA = 0.5;
            int MAX_LS_ITER = 100;
            int pcgmaxi = 5000;
            double eta = 1e-3;

            int pitr = 0;
            
            double lambda = lambda1 * c;
            double t = Math.Min(Math.Max(1.0, 1.0 / lambda), (2 * p / 1e-3));
            double zero = 0;
            double dobj = -1 / zero;
            double pobj;
            double gap;
            double pcgtol;
            double s = 1 / zero;
           



            double err = 0.0;

            double fmax = 0.0;

            //Initializes dimensions for shared memory kernels
            int NgroupSize = Math.Min(this.accelerate.MaxNumThreadsPerGroup, Y2.GetLength(0));
            int Ngridsize = (int)((Y2.GetLength(0) + NgroupSize - 1) / NgroupSize);
            
            int PgroupSize = Math.Min(this.accelerate.MaxNumThreadsPerGroup, X.GetLength(1));
            int Pgridsize = (int)((X.GetLength(1) + PgroupSize - 1) / PgroupSize);
            
            int P2groupSize = Math.Min(this.accelerate.MaxNumThreadsPerGroup, X.GetLength(1) * 2);
            int P2gridsize = (int)(((X.GetLength(1) * 2) + PgroupSize - 1) / PgroupSize);
            
            KernelConfig Ndimension = (
                    Ngridsize, // Compute the number of groups (round up)
                    NgroupSize); 
            KernelConfig Pdimension = (
                    Pgridsize, // Compute the number of groups (round up)
                    PgroupSize); 
            KernelConfig P2dimension = (
                    P2gridsize, // Compute the number of groups (round up)
                    P2groupSize); 

            
            
            




            int PCGj = X.GetLength(1) * 2;
            int PCGn = X.GetLength(1) * 2;
            timeonfunction.Stop();
            Console.WriteLine("Time wasted");
            Console.WriteLine(timeonfunction.Elapsed);
            getasarraytimes.Start();
            using (this.ColMeansBuffer)
            using (this.XBuffer)
            {
                
                //Calculate the mean of each column of X
                columnMeansKern(this.ColMeansBuffer.Extent.ToIntIndex(), this.XBuffer.View, this.ColMeansBuffer.View, X.GetLength(0));
                //Calculate the std dev of each column of X
                columnSTDevKern(this.ColSTDBuffer.Extent.ToIntIndex(), this.XBuffer.View, this.ColSTDBuffer.View, this.ColMeansBuffer.View, X.GetLength(0));
                
                //Fill x2
                x2fillkern(new Index1D(X.GetLength(1)), this.XBuffer.View, this.X2Buffer.View, this.ColMeansBuffer.View, this.ColSTDBuffer.View, c, padding, this.P, X.GetLength(0));
                
                fillInverseMatrixKern(this.X2Buffer.Extent.ToIntIndex(), this.X2Buffer.View, this.X2TransposeBuffer.View);
                
                //Get mean of Y
                columnMeans1DKern(this.YMeanBuffer.Extent.ToIntIndex(), this.YBuffer.View, this.YMeanBuffer.View, Y2.GetLength(0));
                
                //Subtract Y mean from Ybuffer and put into YNorm
                subByColumnsKern(this.YBuffer.Extent.ToIntIndex(), this.YBuffer.View, this.YNormBuffer.View, this.YMeanBuffer.View);
                

                //Initialize U,F,and diagxtx buffers
                setBuffToValueKern(this.UBuffer.Extent.ToIntIndex(), this.UBuffer.View, 1.0);
                FFillKern(this.UBuffer.Extent.ToIntIndex(), this.UBuffer.View, this.WBuffer.View, this.FBuffer.View);
                setBuffToValueKern(this.DiagxtxBuffer.Extent.ToIntIndex(), this.DiagxtxBuffer.View, 2.0);
                
                
                //Prevents O(n^3) kernel from running in case of very large dataset
                if(p < COLLIMIT && n < ROWLIMIT){
                    SMMatMul2DKern(new Index3D(this.X2TransposeBuffer.Extent.ToIntIndex().X, this.X2TransposeBuffer.Extent.ToIntIndex().Y, this.X2Buffer.Extent.ToIntIndex().Y),  this.X2TransposeBuffer.View, this.X2Buffer.View, this.ATABuffer.View);
                }
                
   

                //Begin looping
                for (int niter = 0; niter <= maxIter; niter++)
                {
                    
                    //Get the z values for this iteration multiplyiong x2 by the weights (w)
                    setBuffToValueKern(this.ZBuffer.Extent.ToIntIndex(), this.ZBuffer.View, 0.0);
                    SMMatMul1DKern(this.X2Buffer.Extent.ToIntIndex(),this.X2Buffer.View, this.WBuffer.View, this.ZBuffer.View);

                    //Initialize the nubuffer
                    InitializeNuKern(this.ZBuffer.Extent.ToIntIndex(), this.YNormBuffer.View, this.ZBuffer.View, this.NuBuffer.View);
                    
                    
                    //Set the nubuffer
                    setBuffToValueKern(this.XNuBuffer.Extent.ToIntIndex(), this.XNuBuffer.View, 0.0);
                    //NOTE: When there are more than 65535 rows, the dimensions are too big
                    //If statement to take care of said edge case
                    if(this.X2TransposeBuffer.Extent.ToIntIndex().Y < 65535 ){
                        SMMatMul1DKern(this.X2TransposeBuffer.Extent.ToIntIndex(),this.X2TransposeBuffer.View, this.NuBuffer.View, this.XNuBuffer.View);
                    }
                    else{
                        SMMatMul1DKernTooLarge(this.X2Buffer.Extent.ToIntIndex(),this.X2TransposeBuffer.View, this.NuBuffer.View, this.XNuBuffer.View);
                    }
                    
                    //Get the largest value in xnu
                    GetMaxValKern(this.MaxValBuffer.Extent.ToIntIndex(), this.XNuBuffer.View, this.MaxValBuffer.View, this.XNuBuffer.Extent.ToIntIndex().X);

                    //Calculate the Duality Gap
                    DualityGapKern(this.NuBuffer.Extent.ToIntIndex(), this.NuBuffer.View, this.MaxValBuffer.View, lambda, this.NuBuffer.Extent.ToIntIndex().X);

                    //Clear the Dot Product buffers from previous iterations
                    setBuffToValueKern(this.DotProdBufferZxZ.Extent.ToIntIndex(), this.DotProdBufferZxZ.View, 0.0);
                    setBuffToValueKern(this.DotProdBufferNUxNU.Extent.ToIntIndex(), this.DotProdBufferNUxNU.View, 0.0);
                    setBuffToValueKern(this.DotProdBufferNUxY.Extent.ToIntIndex(), this.DotProdBufferNUxY.View, 0.0);
                    
                    //Executes 3 different dotproducts in the same kernel in order to save runtime.
                    TripleDotKern(Ndimension, this.NuBuffer.View, this.YNormBuffer.View, this.DotProdBufferNUxY.View, this.NuBuffer.View, this.DotProdBufferNUxNU.View,this.ZBuffer.View, this.DotProdBufferZxZ.View,NgroupSize);

                    
                    //Gets the norm value of the weights
                    setBuffToValueKern(this.Norm1Buffer.Extent.ToIntIndex(), this.Norm1Buffer.View, 0.0);
                    SMNormKern(Pdimension, this.WBuffer.View, this.Norm1Buffer.View, PgroupSize);

                    //Calc the Primal and Dual objective function values, used for early stopping
                    PobjKern(this.DotProdBufferZxZ.Extent.ToIntIndex(), this.DotProdBufferZxZ.View, this.Norm1Buffer.View, lambda);
                    pobj = this.Norm1Buffer.GetAsArray1D()[0];
                    DobjKern(this.DotProdBufferNUxNU.Extent.ToIntIndex(), this.DotProdBufferNUxNU, this.DotProdBufferNUxY, dobj);
                    dobj = this.DotProdBufferNUxNU.GetAsArray1D()[0];

                    gap = pobj - dobj;
                    
                    if (niter%10 == 0)
                    {
                        if(verbose){
                            Console.WriteLine("Primal and dual objective function value after {0} iterations: {1}, {2}", niter, pobj, dobj);
                        }
                    }
                    if (gap / dobj < tol)
                    {
                        if(verbose){
                          Console.WriteLine("Primal and dual objective function value after {0} iterations: {1}, {2}", niter, pobj, dobj);
                        }
                        break;
                    }

                    //Adjust s value (based on previous iterations)
                    if (s >= 0.5)
                    {
                        t = Math.Max(Math.Min((2 *p * MU) / gap, MU * t), t);
                    }


                    //Calculate newton step
                    NewtonStepKern(this.UBuffer.Extent.ToIntIndex(), this.UBuffer.View, this.WBuffer.View, this.Q1Buffer.View, this.Q2Buffer.View, this.D1Buffer.View, this.D2Buffer.View, t);

                    //Calculate gradient
                    matrixmulGradphikern(new Index1D(GradphiBuffer.Extent.ToIntIndex().Y), this.X2TransposeBuffer.View, this.ZBuffer.View, this.GradphiBuffer.View);
                    GradPhiKern(this.Q1Buffer.Extent.ToIntIndex(), this.GradphiBuffer.View, this.Q1Buffer.View, this.Q2Buffer.View, this.GradBuffer.View, t, lambda, p);

                    //Calculate the vectors that are used in the preconditioner
                    PreconditionerVectorKern(this.PrbBuffer.Extent.ToIntIndex(), this.PrbBuffer.View, this.PrsBuffer.View, this.D1Buffer.View, this.D2Buffer.View, this.DiagxtxBuffer.View);

                    //Calculate the norm of the gradient
                    setBuffToValueKern(this.GradNormBuffer.Extent.ToIntIndex(), this.GradNormBuffer.View, 0.0);
                    SingleSelfDotKern(P2dimension, this.GradBuffer.View, this.GradNormBuffer.View, P2groupSize);
                    sqrtkern(this.GradNormBuffer.Extent.ToIntIndex(), this.GradNormBuffer.View);

                    //Set the pcgtolerance
                    pcgtol = Math.Min(0.1, (eta * gap) / Math.Min(1.0, this.GradNormBuffer.GetAsArray1D()[0]));

                    if (niter != 0 && pitr == 0)
                    {
                        pcgtol = pcgtol * 0.1;
                    }

                    //Separate cases if p and/or n are too big
                    if(p < COLLIMIT && n < ROWLIMIT){
                        //Fills ataxbuffer with MM ata by dxu
                        setBuffToValueKern(this.ATAXBuffer.Extent.ToIntIndex(), this.ATAXBuffer.View, 0.0);
                        SMMatMul1DKern(this.ATABuffer.Extent.ToIntIndex(),this.ATABuffer.View, this.DXUBuffer.View, this.ATAXBuffer.View);
                    }
                    else{
                        //ATA IS NOT FILLED IN THIS CASE
                        //Fills AXbuffer with MM X2 by dxu, then fills ataxbuffer with MM X2 Transposed by AXbuffer
                        setBuffToValueKern(this.AXBuffer.Extent.ToIntIndex(), this.AXBuffer.View, 0.0);
                        SMMatMul1DKern(this.X2Buffer.Extent.ToIntIndex(),this.X2Buffer.View, this.DXUBuffer.View, this.AXBuffer.View);

                        setBuffToValueKern(this.ATAXBuffer.Extent.ToIntIndex(), this.ATAXBuffer.View, 0.0);
                        if(this.X2TransposeBuffer.Extent.ToIntIndex().Y < 65535 ){
                            SMMatMul1DKern(this.X2TransposeBuffer.Extent.ToIntIndex(),this.X2TransposeBuffer.View, this.AXBuffer.View, this.ATAXBuffer.View);
                        }
                        else{
                            SMMatMul1DKernTooLarge(this.X2Buffer.Extent.ToIntIndex(),this.X2TransposeBuffer.View, this.AXBuffer.View, this.ATAXBuffer.View);
                        }
                    }
                        


                    //Begin preconditioned conjugate gradient

                    //Initialize PCG R buffer
                    PCGmvKern(new Index1D(X2Buffer.Extent.ToIntIndex().Y), this.ATAXBuffer.View, this.DXUBuffer.View, this.PCGBufferR.View, this.D1Buffer.View, this.D2Buffer.View, p);

                    //Then fill R and RR based on gradbuffer values
                    RandRRFillKern(this.PCGBufferR.Extent.ToIntIndex(), this.GradBuffer.View, this.PCGBufferR.View, this.PCGBufferRR.View);




                    //Calculate the norm of gradbuffer
                    setBuffToValueKern(this.BNormBuffer.Extent.ToIntIndex(), this.BNormBuffer.View, 0.0);
                    SingleSelfDotKern(P2dimension, this.GradBuffer.View, this.BNormBuffer.View, P2groupSize);
                    sqrtkern(this.BNormBuffer.Extent.ToIntIndex(), this.BNormBuffer.View);

                    //Asolve on r,z, this edits Z.
                    PCGasolveKern(new Index1D(X.GetLength(1)), this.PCGBufferR.View, this.PCGBufferZ.View, this.D1Buffer.View, this.D2Buffer.View, this.PrsBuffer.View, this.PrbBuffer.View, p);


                    for (int iter = 1; iter < pcgmaxi; iter++)
                    {
                        //Asolve on rr,zz, this edits ZZ.
                        PCGasolveKern(new Index1D(X.GetLength(1)), this.PCGBufferRR.View, this.PCGBufferZZ.View, this.D1Buffer.View, this.D2Buffer.View, this.PrsBuffer.View, this.PrbBuffer.View, p);


                        //Calculate BKNUM value
                        setBuffToValueKern(this.BKNumBuffer.Extent.ToIntIndex(), this.BKNumBuffer.View, 0.0);
                        SingleDotKern(P2dimension, this.PCGBufferRR.View, this.PCGBufferZ.View, this.BKNumBuffer.View,P2groupSize);

                        

                        //Fill PCG p and pp buffers
                        if (iter == 1)
                        {
                            //Set equal to z and zz
                            FirstIterationFillPsKern(this.PCGBufferP.Extent.ToIntIndex(), this.PCGBufferP.View, this.PCGBufferPP.View, this.PCGBufferZ.View, this.PCGBufferZZ.View);
                        }
                        else
                        {   
                            //multiply p & pp by (bknum/bkden), and then add z and zz
                            FillPsKern(this.PCGBufferP.Extent.ToIntIndex(), this.PCGBufferP.View, this.PCGBufferPP.View, this.PCGBufferZ.View, this.PCGBufferZZ.View, this.BKNumBuffer.View, this.BKDenBuffer.View);

                        }
                        //Set bkden to bknum
                        CopyBufferKern1D(this.BKNumBuffer.Extent.ToIntIndex(), this.BKNumBuffer.View, this.BKDenBuffer.View);

                        //Accounts for if ATA has been initialized or not
                        if(p < COLLIMIT && n < ROWLIMIT){
                            setBuffToValueKern(this.ATAXBuffer.Extent.ToIntIndex(), this.ATAXBuffer.View, 0.0);
                            SMMatMul1DKern(this.ATABuffer.Extent.ToIntIndex(),this.ATABuffer.View, this.PCGBufferP.View, this.ATAXBuffer.View);
                        }
                        else{
                            setBuffToValueKern(this.AXBuffer.Extent.ToIntIndex(), this.AXBuffer.View, 0.0);
                            SMMatMul1DKern(this.X2Buffer.Extent.ToIntIndex(),this.X2Buffer.View, this.PCGBufferP.View, this.AXBuffer.View);

                            setBuffToValueKern(this.ATAXBuffer.Extent.ToIntIndex(), this.ATAXBuffer.View, 0.0);
                            if(this.X2TransposeBuffer.Extent.ToIntIndex().Y < 65535 ){
                                SMMatMul1DKern(this.X2TransposeBuffer.Extent.ToIntIndex(),this.X2TransposeBuffer.View, this.AXBuffer.View, this.ATAXBuffer.View);
                            }
                            else{
                                SMMatMul1DKernTooLarge(this.X2Buffer.Extent.ToIntIndex(),this.X2TransposeBuffer.View, this.AXBuffer.View, this.ATAXBuffer.View);
                            }
                        }

                        //mv(p,z)
                        PCGmvKern(new Index1D(X2Buffer.Extent.ToIntIndex().Y), this.ATAXBuffer.View, this.PCGBufferP.View, this.PCGBufferZ.View, this.D1Buffer.View, this.D2Buffer.View, p);

                        //Calculates AKDEN value
                        setBuffToValueKern(this.AKDenBuffer.Extent.ToIntIndex(), this.AKDenBuffer.View, 0.0);
                        SingleDotKern(P2dimension, this.PCGBufferPP.View, this.PCGBufferZ.View, this.AKDenBuffer.View,P2groupSize);

                        
                        //Accounts for if ATA has been initialized or not
                        if(p < COLLIMIT && n < ROWLIMIT){
                            setBuffToValueKern(this.ATAXBuffer.Extent.ToIntIndex(), this.ATAXBuffer.View, 0.0);
                            SMMatMul1DKern(this.ATABuffer.Extent.ToIntIndex(),this.ATABuffer.View, this.PCGBufferPP.View, this.ATAXBuffer.View);
                        }
                        else{
                            setBuffToValueKern(this.AXBuffer.Extent.ToIntIndex(), this.AXBuffer.View, 0.0);
                            SMMatMul1DKern(this.X2Buffer.Extent.ToIntIndex(),this.X2Buffer.View, this.PCGBufferPP.View, this.AXBuffer.View);

                            setBuffToValueKern(this.ATAXBuffer.Extent.ToIntIndex(), this.ATAXBuffer.View, 0.0);
                            if(this.X2TransposeBuffer.Extent.ToIntIndex().Y < 65535 ){
                                SMMatMul1DKern(this.X2TransposeBuffer.Extent.ToIntIndex(),this.X2TransposeBuffer.View, this.AXBuffer.View, this.ATAXBuffer.View);
                            }
                            else{
                                SMMatMul1DKernTooLarge(this.X2Buffer.Extent.ToIntIndex(),this.X2TransposeBuffer.View, this.AXBuffer.View, this.ATAXBuffer.View);
                            }
                        }

                        //mv(pp,zz)
                        PCGmvKern(new Index1D(X2Buffer.Extent.ToIntIndex().Y), this.ATAXBuffer.View, this.PCGBufferPP.View, this.PCGBufferZZ.View, this.D1Buffer.View, this.D2Buffer.View, p);
                        
                        
                        //Adjusts dxu, r and rr 
                        SubFromRsKern(this.PCGBufferR.Extent.ToIntIndex(), this.PCGBufferR.View, this.PCGBufferRR.View, this.PCGBufferZ.View, this.PCGBufferZZ.View, this.BKNumBuffer.View, this.AKDenBuffer.View, this.DXUBuffer.View, this.PCGBufferP.View);
                        
                        //asolve(r,z)
                        PCGasolveKern(new Index1D(X.GetLength(1)), this.PCGBufferR.View, this.PCGBufferZ.View, this.D1Buffer.View, this.D2Buffer.View, this.PrsBuffer.View, this.PrbBuffer.View, p);


                        //Calculates rnorm
                        setBuffToValueKern(this.RNormBuffer.Extent.ToIntIndex(), this.RNormBuffer.View, 0.0);
                        SingleSelfDotKern(P2dimension, this.PCGBufferR.View, this.RNormBuffer.View, P2groupSize);
                        sqrtkern(this.RNormBuffer.Extent.ToIntIndex(), this.RNormBuffer.View);
                        

                        //error = rnorm/bnorm
                        calcErrorKern(this.ErrBuffer.Extent.ToIntIndex(), this.RNormBuffer.View, this.BNormBuffer.View, this.ErrBuffer.View);
                        err = this.ErrBuffer.GetAsArray1D()[0];
                        
                        if (iter % 10 == 0)
                        {
                            if(verbose){
                                Console.WriteLine("BCG: Error after {0} iterations: {1}", iter, err);
                            }
                        }
                        if (err <= pcgtol)
                        {
                            if(verbose){
                                Console.WriteLine("BCG END: Error after {0} iterations: {1}", iter, err);
                            }
                            break;
                        }
                    }
                    
                    if (err > pcgtol)
                    {
                        pitr = pcgmaxi;
                    }


                    //Fill dx and du
                    splitDXUKern(this.DXBuffer.Extent.ToIntIndex(), this.DXBuffer.View, this.DUBuffer.View, this.DXUBuffer.View, this.DXBuffer.Extent.ToIntIndex().X);


                    
                    //Get dot product of zxz
                    setBuffToValueKern(this.DotProdBufferZxZ.Extent.ToIntIndex(), this.DotProdBufferZxZ.View, 0.0);
                    SingleSelfDotKern(Ndimension, this.ZBuffer.View, this.DotProdBufferZxZ.View,NgroupSize);

                    //Get sum of U
                    setBuffToValueKern(this.USumBuffer.Extent.ToIntIndex(), this.USumBuffer.View, 0.0);
                    SMSumOfKern(Pdimension, this.UBuffer.View, this.USumBuffer.View, PgroupSize);
                   

                    //Get the sum of the negative log of all values in f
                    setBuffToValueKern(this.FSumlgBuffer.Extent.ToIntIndex(), this.FSumlgBuffer.View, 0.0);
                    SMSumLogNegKern(Pdimension,  this.FBuffer.View, this.FSumlgBuffer.View, PgroupSize);

                    //Calculates phi
                    CalcphiKern(this.PhiBuffer.Extent.ToIntIndex(), this.PhiBuffer.View, this.DotProdBufferZxZ.View, this.USumBuffer.View, this.FSumlgBuffer.View, t, lambda);

                    //Gets the dot product of GradxDxu
                    setBuffToValueKern(this.DotProdBufferGradxDXU.Extent.ToIntIndex(), this.DotProdBufferGradxDXU.View, 0.0);
                    SingleDotKern(P2dimension, this.GradBuffer.View, this.DXUBuffer.View, this.DotProdBufferGradxDXU.View,P2groupSize);



                    //init s
                    s = 1.0;
                    
                    for (int lsiter = 0; lsiter < MAX_LS_ITER; lsiter++)
                    {

                        
                        setBuffToValueKern(this.FMaxBuffer.Extent.ToIntIndex(), this.FMaxBuffer.View, -1.0);
                        //Sets neww, newu, newf, and flips fmax to positive if any value of newf is positive
                        setnewBuffersKern(this.WBuffer.Extent.ToIntIndex(), this.WBuffer.View, this.UBuffer.View, this.DXBuffer.View, this.DUBuffer.View, this.NewWBuffer.View, this.NewUBuffer.View, this.NewFBuffer.View, this.FMaxBuffer.View, s);

                        
                        fmax = FMaxBuffer.GetAsArray1D()[0];
                        if (fmax < 0.0)
                        {
                            
                            //Calculates new z
                            setBuffToValueKern(this.NewZBuffer.Extent.ToIntIndex(), this.NewZBuffer.View, 0.0);
                            SMMatMul1DKern(this.X2Buffer.Extent.ToIntIndex(),this.X2Buffer.View, this.NewWBuffer.View, this.NewZBuffer.View);

                            //Subs from z the corresponding y norms
                            SubYFromZKern(this.NewZBuffer.Extent.ToIntIndex(), this.NewZBuffer.View, this.YNormBuffer.View);

                            //Gets new dot product of newz x newz
                            setBuffToValueKern(this.DotProdBufferZxZ.Extent.ToIntIndex(), this.DotProdBufferZxZ.View, 0.0);
                            SingleSelfDotKern(Ndimension, this.NewZBuffer.View, this.DotProdBufferZxZ.View,NgroupSize);


                            //Gets NewUSUM
                            setBuffToValueKern(this.USumBuffer.Extent.ToIntIndex(), this.USumBuffer.View, 0.0);
                            SMSumOfKern(Pdimension, this.NewUBuffer.View, this.USumBuffer.View, PgroupSize);

                            //Get the sum of the negative log of all values in newf
                            setBuffToValueKern(this.FSumlgBuffer.Extent.ToIntIndex(), this.FSumlgBuffer.View, 0.0);
                            SMSumLogNegKern(Pdimension,  this.NewFBuffer.View, this.FSumlgBuffer.View, PgroupSize);


                            //Calculate the newphie
                            CalcphiKern(this.NewPhiBuffer.Extent.ToIntIndex(), this.NewPhiBuffer.View, this.DotProdBufferZxZ.View, this.USumBuffer.View, this.FSumlgBuffer.View, t, lambda);

                            //Compare difference of newphi and oldphi to Alpha, s, GradXDxu
                            nphicompkern(new Index1D(1), this.PhiBuffer.View, this.NewPhiBuffer.View, this.DotProdBufferGradxDXU.View, this.CompPhiBuffer.View, ALPHA, s);

                            if (this.CompPhiBuffer.GetAsArray1D()[0] == 1)
                            {
                                
                                break;
                            }
                        }
                        //Adjust s each iteration
                        s = BETA * s;
                        


                    }
                    //Set w u and f to neww, newu, newf
                    CopyBufferKern1D(this.NewWBuffer.Extent.ToIntIndex(), this.NewWBuffer.View, this.WBuffer.View);
                    CopyBufferKern1D(this.NewUBuffer.Extent.ToIntIndex(), this.NewUBuffer.View, this.UBuffer.View);
                    CopyBufferKern2D(this.NewFBuffer.Extent.ToIntIndex(), this.NewFBuffer.View, this.FBuffer.View);

                   


                }
                
                setBuffToValueKern(this.DotProdBufferWxCenter.Extent.ToIntIndex(), this.DotProdBufferWxCenter.View, 0.0);
                //Does final adjustment to weights
                WFinaleKern(this.WBuffer.Extent.ToIntIndex(), this.WBuffer.View, this.ColSTDBuffer.View, c);
                SingleDotKern(Pdimension, this.WBuffer.View, this.ColMeansBuffer.View, this.DotProdBufferWxCenter.View,PgroupSize);

                //Calc b value, used for predictions
                this.B = ymeansum/Y.GetLength(0) - DotProdBufferWxCenter.GetAsArray1D()[0];
           

                this.W = this.WBuffer.GetAsArray1D();
                

                
            }
            getasarraytimes.Stop();
            
            Console.WriteLine("GET AS ARRAY TIMES");
            Console.WriteLine(getasarraytimes.Elapsed);
            Console.WriteLine("timeonfunction");

            Console.WriteLine(timeonfunction.Elapsed);
            return this;


        }
        static void TripleDotKernal(
            ArrayView<double> aView,
            ArrayView<double> bView,
            ArrayView<double> Output,
            ArrayView<double> aView2,
            ArrayView<double> Output2,
            ArrayView<double> aView3,
            ArrayView<double> Output3,
            int gridsize)        
        {
            ///<summary>Calculates 3 different dot products</summary>
            ///<param name="aView">The first view of the first dot product</param>
            ///<param name="bView">The second view of the first dot product</param>
            ///<param name="Output">The first dot product</param>
            ///<param name="aView2">The view used for the second dot product</param>
            ///<param name="Output2">The dot product of aView2 * aView2</param>
            ///<param name="aView3">The view used for the third dot product</param>
            ///<param name="Output3">The dot product of aView3 * aView3</param>
            ///<param name="gridsize">The number of elements in each grid</param>
            

            int globalIndex = Grid.LinearIndex;
            
            int localindex = Group.LinearIndex;

            // 'Allocate' a single shared memory variable of type int (= 4 bytes)
            ref double sharedVariable = ref ILGPU.SharedMemory.Allocate<double>();
            // ref int sharedIndex = ref ILGPU.SharedMemory.Allocate<int>();
            ref double sharedVariable2 = ref ILGPU.SharedMemory.Allocate<double>();
            ref double sharedVariable3 = ref ILGPU.SharedMemory.Allocate<double>();
            double val = 0.0;
            double val2 = 0.0;
            double val3 = 0.0;
            // Initialize shared memory
            if (Group.IsFirstThread)
                sharedVariable = 0;
                sharedVariable2 = 0;
                sharedVariable3 = 0;
                

            Group.Barrier();
            
            if (globalIndex*gridsize + localindex < aView.Length){
                val = aView[globalIndex*gridsize + localindex] *bView[globalIndex*gridsize + localindex];
                Atomic.Add(ref sharedVariable, val);
                val2 = aView2[globalIndex*gridsize + localindex] * aView2[globalIndex*gridsize + localindex];
                Atomic.Add(ref sharedVariable2, val2);
                val3 = aView3[globalIndex*gridsize + localindex] * aView3[globalIndex*gridsize + localindex];
                Atomic.Add(ref sharedVariable3, val3);
            }
           
            Group.Barrier();   
            
            if (Group.IsLastThread){

                Atomic.Add(ref Output[0], sharedVariable);
                Atomic.Add(ref Output2[0], sharedVariable2);    
                Atomic.Add(ref Output3[0], sharedVariable3);        
            }   
            
            
            
        }
        static void SingleDotKernal(
            ArrayView<double> aView,
            ArrayView<double> bView,
            ArrayView<double> Output,
            int gridsize)        
        {
            ///<summary>Calculates the dot of two arrayViews</summary>
            ///<param name="aView">The first view of the dot product</param>
            ///<param name="bView">The second view of the dot product</param>
            ///<param name="Output">The first dot product</param>
            ///<param name="gridsize">The number of elements in each grid</param>


            int globalIndex = Grid.LinearIndex;
            
            int localindex = Group.LinearIndex;

            // 'Allocate' a single shared memory variable of type int (= 4 bytes)
            ref double sharedVariable = ref ILGPU.SharedMemory.Allocate<double>();
            // ref int sharedIndex = ref ILGPU.SharedMemory.Allocate<int>();
        
            double val = 0.0;
           
            // Initialize shared memory
            if (Group.IsFirstThread)
                sharedVariable = 0;
                
                

            Group.Barrier();
            
            if (globalIndex*gridsize + localindex < aView.Length){
                val = aView[globalIndex*gridsize + localindex] *bView[globalIndex*gridsize + localindex];
                Atomic.Add(ref sharedVariable, val);
                
            }
           
            Group.Barrier();   
            
            if (Group.IsLastThread){

                Atomic.Add(ref Output[0], sharedVariable);
                
            }  
            

            
            
        }
        static void SMNormKernal(
            ArrayView<double> aView,
            ArrayView<double> Output,
            int gridsize)        
        {
            ///<summary>Calculates the norm1 of an arrayView</summary>
            ///<param name="aView">The ArrayView</param>
            ///<param name="Output">The norm1 of aView</param>
            ///<param name="gridsize">The number of elements in each grid</param>

            int globalIndex = Grid.LinearIndex;
            
            int localindex = Group.LinearIndex;

            // 'Allocate' a single shared memory variable of type int (= 4 bytes)
            ref double sharedVariable = ref ILGPU.SharedMemory.Allocate<double>();
            // ref int sharedIndex = ref ILGPU.SharedMemory.Allocate<int>();
        
            double val = 0.0;
           
            // Initialize shared memory
            if (Group.IsFirstThread)
                sharedVariable = 0;
                
                

            Group.Barrier();
            
            if (globalIndex*gridsize + localindex < aView.Length){
                val = Math.Abs(aView[globalIndex*gridsize + localindex]);
                Atomic.Add(ref sharedVariable, val);
                
            }
           
            Group.Barrier();   
            
            if (Group.IsLastThread){

                Atomic.Add(ref Output[0], sharedVariable);
                
            }
        }
        static void SingleSelfDotKernal(
            ArrayView<double> aView,
            ArrayView<double> Output,
            int gridsize)        
        {
            ///<summary>Calculates the dot product of an arrayView against itself</summary>
            ///<param name="aView">The view of the dot product</param>
            ///<param name="Output">The first dot product</param>
            ///<param name="gridsize">The number of elements in each grid</param>

            int globalIndex = Grid.LinearIndex;
            
            int localindex = Group.LinearIndex;

            // 'Allocate' a single shared memory variable of type int (= 4 bytes)
            ref double sharedVariable = ref ILGPU.SharedMemory.Allocate<double>();
            // ref int sharedIndex = ref ILGPU.SharedMemory.Allocate<int>();
        
            double val = 0.0;
           
            // Initialize shared memory
            if (Group.IsFirstThread)
                sharedVariable = 0;
                
                

            Group.Barrier();
            
            if (globalIndex*gridsize + localindex < aView.Length){
                val = aView[globalIndex*gridsize + localindex] *aView[globalIndex*gridsize + localindex];
                Atomic.Add(ref sharedVariable, val);
                
            }
           
            Group.Barrier();   
            
            if (Group.IsLastThread){

                Atomic.Add(ref Output[0], sharedVariable);
                
            }  
           

            
            
        }
        
        static void SharedMemoryArrayKernel(
            ArrayView<double> dataView,     // A view to a chunk of memory (1D in this case)
            ArrayView<double> outputView)   // A view to a chunk of memory (1D in this case)
        {
            // Compute the global 1D index for accessing the data view
            int globalIndex = Grid.GlobalIndex.X;

            // Declares a shared-memory array with 128 elements of type int = 4 * 128 = 512 bytes
            // of shared memory per group
            // Note that 'Allocate' requires a compile-time known constant array size.
            // If the size is unknown at compile-time, consider using `GetDynamic`.
            ref double sharedVariable = ref ILGPU.SharedMemory.Allocate<double>();


            if (Group.IsFirstThread)
                sharedVariable = 0;
            // Load the element into shared memory
            // var value = globalIndex < dataView.Length ?
            //     dataView[globalIndex] :
            //     0;
            // sharedArray[Group.IdxX] = value;

            // Wait for all threads to complete the loading process
            Group.Barrier();
            Atomic.Add(ref sharedVariable, dataView[globalIndex]);

            // Compute the sum over all elements in the group
            // double sum = 0;
            // for (int i = 0, e = Group.Dimension.X; i < e; ++i)
            //     sum += sharedArray[i];
            Group.Barrier();
            // Store the sum
            //if (globalIndex < outputView.Length)
            Atomic.Add(ref outputView[0],  dataView[globalIndex]);
        }
        static void SharedMemoryVariableKerneL(
            ArrayView<double> dataView,          // A view to a chunk of memory (1D in this case)
            ArrayView<double> outputView,
            int gridsize, int remainder)        // A view to a chunk of memory (1D in this case)
        {
            //Compute the global 1D index for accessing the data view
            int globalIndex = Grid.LinearIndex;
            
            int localindex = Group.LinearIndex;

            // 'Allocate' a single shared memory variable of type int (= 4 bytes)
            ref double sharedVariable = ref ILGPU.SharedMemory.Allocate<double>();
            // ref int sharedIndex = ref ILGPU.SharedMemory.Allocate<int>();
            // ref int sharedIndex2 = ref ILGPU.SharedMemory.Allocate<int>();
            // ref int sharedIndex3 = ref ILGPU.SharedMemory.Allocate<int>();

            // Initialize shared memory
            if (Group.IsFirstThread)
                 sharedVariable = 0;
                 
            Group.Barrier();
            
            if (globalIndex*gridsize + localindex < dataView.Length){
                Atomic.Add(ref sharedVariable, dataView[globalIndex*gridsize + localindex]);
            }
           
            Group.Barrier();   
            
            if (Group.IsLastThread){

                Atomic.Add(ref outputView[0], sharedVariable);    
            }   
           

            
        }
        static void SMSumOfKernal(
            ArrayView<double> dataView,          // A view to a chunk of memory (1D in this case)
            ArrayView<double> outputView,
            int gridsize)        // A view to a chunk of memory (1D in this case)
        {
            //Compute the global 1D index for accessing the data view
            int globalIndex = Grid.LinearIndex;
            
            int localindex = Group.LinearIndex;

            // 'Allocate' a single shared memory variable of type int (= 4 bytes)
            ref double sharedVariable = ref ILGPU.SharedMemory.Allocate<double>();
            // ref int sharedIndex = ref ILGPU.SharedMemory.Allocate<int>();
            // ref int sharedIndex2 = ref ILGPU.SharedMemory.Allocate<int>();
            // ref int sharedIndex3 = ref ILGPU.SharedMemory.Allocate<int>();

            // Initialize shared memory
            if (Group.IsFirstThread)
                 sharedVariable = 0;
                 
            Group.Barrier();
            
            if (globalIndex*gridsize + localindex < dataView.Length){
                Atomic.Add(ref sharedVariable, dataView[globalIndex*gridsize + localindex]);
            }
           
            Group.Barrier();   
            
            if (Group.IsLastThread){

                Atomic.Add(ref outputView[0], sharedVariable);    
            }   
           

            
        }
        static void SMSumLogNegKernal(
            ArrayView2D<double, Stride2D.DenseX> dataView,          // A view to a chunk of memory (1D in this case)
            ArrayView<double> outputView,
            int gridsize)        // A view to a chunk of memory (1D in this case)
        {
            //Compute the global 1D index for accessing the data view
            int globalIndex = Grid.LinearIndex;
            
            int localindex = Group.LinearIndex;

            // 'Allocate' a single shared memory variable of type int (= 4 bytes)
            ref double sharedVariable = ref ILGPU.SharedMemory.Allocate<double>();
            // ref int sharedIndex = ref ILGPU.SharedMemory.Allocate<int>();
            // ref int sharedIndex2 = ref ILGPU.SharedMemory.Allocate<int>();
            // ref int sharedIndex3 = ref ILGPU.SharedMemory.Allocate<int>();
            double val =0.0;
            // Initialize shared memory
            if (Group.IsFirstThread)
                 sharedVariable = 0;
                 
            Group.Barrier();
            
            if (globalIndex*gridsize + localindex < dataView.Length){
                val = Math.Log(-1.0 *dataView[new Index2D(0, globalIndex*gridsize + localindex)]);
                Atomic.Add(ref sharedVariable, val);
                val = Math.Log( -1.0 *dataView[new Index2D(1, globalIndex*gridsize + localindex)]);
                Atomic.Add(ref sharedVariable, val);
            }
           
            Group.Barrier();   
            
            if (Group.IsLastThread){

                Atomic.Add(ref outputView[0], sharedVariable);    
            }   
           

            
        }
        static void nphicompkernal(Index1D index, 
            ArrayView1D<double, Stride1D.Dense> phi,
            ArrayView1D<double, Stride1D.Dense> newphi,
            ArrayView1D<double, Stride1D.Dense> gradxDxu,
            ArrayView1D<int, Stride1D.Dense> output,
            double alpha,
            double s){
            if(newphi[index] - phi[index] <= alpha * s * gradxDxu[index]){
                output[index] = 1;

            }
            else{
                output[index] = 0;
            }
            //nphicomp = NewPhiBuffer.GetAsArray1D()[0] - PhiBuffer.GetAsArray1D()[0] <= ALPHA * s * DotProdBufferGradxDXU.GetAsArray1D()[0]

        }
        static void testKernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> aView,
            double c
            )
        {
            for(int i = 0; i < 1000; i ++){
                aView[index] += aView[index] * c;
            }
            
        }
        static void WFinaleKernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> wView,
            ArrayView1D<double, Stride1D.Dense> scale,
            double c
            )
        {
            wView[index] = (c * wView[index]) / scale[index];

        }
        static void SubYFromZKernal(
            Index1D index,
            ArrayView1D<double, Stride1D.Dense> zView,
            ArrayView1D<double, Stride1D.Dense> YView)
        {
            zView[index] = zView[index] - YView[index];


        }

        //Not put in yet
        static void fmaxKernal(Index1D index,
            ArrayView2D<double, Stride2D.DenseX> fView,
            ArrayView1D<double, Stride1D.Dense> maxView,
            int dim1, int dim2)
        {
            maxView[index] = fView[new Index2D(0, 0)];
            for (int i = 0; i < dim1; i++)
            {
                for (int j = 0; j < dim2; j++)
                {
                    if (maxView[index] < fView[new Index2D(i, j)])
                    {
                        maxView[index] = fView[new Index2D(i, j)];
                    }
                }
            }

        }
        
        static void CalcphiKernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> phi,
            ArrayView1D<double, Stride1D.Dense> zdot,
            ArrayView1D<double, Stride1D.Dense> usum,
            ArrayView1D<double, Stride1D.Dense> fsumlgneg,
            double t, double lambda)
        {
            phi[index] = zdot[index] + lambda * usum[index] - fsumlgneg[index] / t;

        }
        static void setnewBuffersKernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> wView,
            ArrayView1D<double, Stride1D.Dense> uView,
            ArrayView1D<double, Stride1D.Dense> dxView,
            ArrayView1D<double, Stride1D.Dense> duView,
            ArrayView1D<double, Stride1D.Dense> newwView,
            ArrayView1D<double, Stride1D.Dense> newuView,
            ArrayView2D<double, Stride2D.DenseX> newfView,
            ArrayView1D<double, Stride1D.Dense> fmax,
            double s)
        {
            newwView[index] = wView[index] + s * dxView[index];
            newuView[index] = uView[index] + s * duView[index];
            newfView[new Index2D(0, index.X)] = newwView[index] - newuView[index];
            newfView[new Index2D(1, index.X)] = (-1.0 * newwView[index]) - newuView[index];
            if(newfView[new Index2D(0, index.X)] > 0 || newfView[new Index2D(1, index.X)] > 0){
                fmax[0] = 1.0;
            }

        }

        //Need to implement
        //Above not initialized yet
        static void sumlognegKernal(Index1D index,
            ArrayView2D<double, Stride2D.DenseX> fView,
            ArrayView1D<double, Stride1D.Dense> sumView,
            int dim1, int dim2)
        {
            for (int i = 0; i < dim1; i++)
            {
                for (int j = 0; j < dim2; j++)
                {     
                    sumView[index] +=Math.Log(-fView[new Index2D(i, j)]);

                }
            }

        }

        static void splitDXUKernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> dxView,
            ArrayView1D<double, Stride1D.Dense> duView,
            ArrayView1D<double, Stride1D.Dense> dxuView,
            int p)
        {
            dxView[index] = dxuView[index];
            duView[index] = dxuView[new Index1D(index.X + p)];

        }
        static void calcErrorKernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> rnorm,
            ArrayView1D<double, Stride1D.Dense> bnorm,
            ArrayView1D<double, Stride1D.Dense> error
            )
        {
            error[index] = rnorm[index] / bnorm[index];

        }

        static void SubFromRsKernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> rView,
            ArrayView1D<double, Stride1D.Dense> rrView,
            ArrayView1D<double, Stride1D.Dense> zView,
            ArrayView1D<double, Stride1D.Dense> zzView,
            ArrayView1D<double, Stride1D.Dense> bknumView,
            ArrayView1D<double, Stride1D.Dense> akdenView,
            ArrayView1D<double, Stride1D.Dense> xView,
            ArrayView1D<double, Stride1D.Dense> pView)
        {
            rView[index] -= (bknumView[new Index1D(0)] / akdenView[new Index1D(0)]) * zView[index];
            rrView[index] -= (bknumView[new Index1D(0)] / akdenView[new Index1D(0)]) * zzView[index];
            xView[index] += (bknumView[new Index1D(0)] / akdenView[new Index1D(0)]) * pView[index];

        }

        static void CopyBufferKernal1D(Index1D index,
            ArrayView1D<double, Stride1D.Dense> aView,
            ArrayView1D<double, Stride1D.Dense> bView)
        {
            bView[index] = aView[index];

        }
        static void CopyBufferKernal2D(Index2D index,
            ArrayView2D<double, Stride2D.DenseX> aView,
            ArrayView2D<double, Stride2D.DenseX> bView)
        {
            bView[index] = aView[index];

        }

        //The two use the same
        static void BKNUM_AKDENKernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> bknumView,
            ArrayView1D<double, Stride1D.Dense> rrView,
            ArrayView1D<double, Stride1D.Dense> zView,
            int n)
        {
            for (int i = 0; i < n; i++)
            {
                bknumView[index] += zView[new Index1D(i)] * rrView[new Index1D(i)];
            }

        }

        static void FirstIterationFillPsKernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> pView,
            ArrayView1D<double, Stride1D.Dense> ppView,
            ArrayView1D<double, Stride1D.Dense> zView,
            ArrayView1D<double, Stride1D.Dense> zzView)
        {
            pView[index] = zView[index];
            ppView[index] = zzView[index];

        }

        static void FillPsKernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> pView,
            ArrayView1D<double, Stride1D.Dense> ppView,
            ArrayView1D<double, Stride1D.Dense> zView,
            ArrayView1D<double, Stride1D.Dense> zzView,
            ArrayView1D<double, Stride1D.Dense> bknumView,
            ArrayView1D<double, Stride1D.Dense> bkdenView)
        {

            pView[index] = ((bknumView[new Index1D(0)] / bkdenView[new Index1D(0)]) * pView[index]) + zView[index];
            ppView[index] = ((bknumView[new Index1D(0)] / bkdenView[new Index1D(0)]) * ppView[index]) + zzView[index];

        }


        static void fillInverseMatrixKernal(Index2D index,
            ArrayView2D<double, Stride2D.DenseX> aView,
            ArrayView2D<double, Stride2D.DenseX> inverseView)
        {
            inverseView[new Index2D(index.Y, index.X)] = aView[index];

        }
        static void PCGasolveKernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> bView,
            ArrayView1D<double, Stride1D.Dense> xView,
            ArrayView1D<double, Stride1D.Dense> d1View,
            ArrayView1D<double, Stride1D.Dense> d2View,
            ArrayView1D<double, Stride1D.Dense> prsView,
            ArrayView1D<double, Stride1D.Dense> prbView,
            int p
            )
        {
            xView[index] = ((d1View[index] * bView[index]) - (d2View[index] * bView[new Index1D(index.X + p)])) / prsView[index];
            xView[new Index1D(index.X + p)] = ((-d2View[index] * bView[index]) + (prbView[index] * bView[new Index1D(index.X + p)])) / prsView[index];

        }
        static void RandRRFillKernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> bView,
            ArrayView1D<double, Stride1D.Dense> rView,
            ArrayView1D<double, Stride1D.Dense> rrView
            )
        {
            rView[index] = bView[index] - rView[index];
            rrView[index] = rView[index];
        }
        //need to initialize ata first, and matrix multiply by AtA x x -> atax
        //Index needs to be passed as p (aka 1/2 of the len of xView)
        static void PCGmvKernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> ataxView,
            ArrayView1D<double, Stride1D.Dense> xView,
            ArrayView1D<double, Stride1D.Dense> yView,
            ArrayView1D<double, Stride1D.Dense> d1View,
            ArrayView1D<double, Stride1D.Dense> d2View,
            int p
            )
        {
            int z = index.X;
            //Issue here, idk why
            yView[index] = (2.0 * ataxView[index]) + (d1View[index] * xView[index]) + (d2View[index] * xView[new Index1D(index.X + p)]);
            yView[new Index1D(index.X + p)] = (d2View[index] * xView[index]) + (d1View[index] * xView[new Index1D(index.X + p)]);



        }
        static void GradPhiAndPreconditionerKernal(Index1D index,
            ArrayView2D<double, Stride2D.DenseX> GradPhiView,
            ArrayView1D<double, Stride1D.Dense> q1View,
            ArrayView1D<double, Stride1D.Dense> q2View,
            ArrayView1D<double, Stride1D.Dense> gradView,
            double t,
            double lambda,
            int p,
            ArrayView1D<double, Stride1D.Dense> prbView,
            ArrayView1D<double, Stride1D.Dense> prsView,
            ArrayView1D<double, Stride1D.Dense> d1View,
            ArrayView1D<double, Stride1D.Dense> d2View,
            ArrayView1D<double, Stride1D.Dense> diagxtxView
            )
        {
            GradPhiView[new Index2D(0, index.X)] = (2.0 * GradPhiView[new Index2D(0, index.X)]) - ((q1View[index] - q2View[index]) / t);
            GradPhiView[new Index2D(1, index.X)] = lambda - (q1View[index] + q2View[index]) / t;//
            gradView[index] = -1.0 * GradPhiView[new Index2D(0, index.X)];
            gradView[new Index1D(index.X + p)] = -1.0 * GradPhiView[new Index2D(1, index.X)];
            prbView[index] = diagxtxView[index] + d1View[index];
            prsView[index] = (prbView[index] * d1View[index]) - (d2View[index] * d2View[index]);

        }
        static void PreconditionerVectorKernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> prbView,
            ArrayView1D<double, Stride1D.Dense> prsView,
            ArrayView1D<double, Stride1D.Dense> d1View,
            ArrayView1D<double, Stride1D.Dense> d2View,
            ArrayView1D<double, Stride1D.Dense> diagxtxView)
        {
            prbView[index] = diagxtxView[index] + d1View[index];
            prsView[index] = (prbView[index] * d1View[index]) - (d2View[index] * d2View[index]);

        }
        static void GradPhiKernal(Index1D index,
            ArrayView2D<double, Stride2D.DenseX> GradPhiView,
            ArrayView1D<double, Stride1D.Dense> q1View,
            ArrayView1D<double, Stride1D.Dense> q2View,
            ArrayView1D<double, Stride1D.Dense> gradView,
            double t,
            double lambda,
            int p
            )
        {
            GradPhiView[new Index2D(0, index.X)] = (2.0 * GradPhiView[new Index2D(0, index.X)]) - ((q1View[index] - q2View[index]) / t);
            GradPhiView[new Index2D(1, index.X)] = lambda - (q1View[index] + q2View[index]) / t;//
            gradView[index] = -1.0 * GradPhiView[new Index2D(0, index.X)];
            gradView[new Index1D(index.X + p)] = -1.0 * GradPhiView[new Index2D(1, index.X)];

        }
        static void sumofkernal(Index1D index, ArrayView1D<double, Stride1D.Dense> aView, ArrayView1D<double, Stride1D.Dense> sum, int dim1)
        {
            ///<summary>Adds up everything in a 2DBuffer.</summary>
            ///<param name="index">(Index1D) Index to iterate through the array</param>
            ///<param name="aView">(ArrayView2D<double, Stride2D.DenseX>) Buffer to be added up</param>
            ///<param name="sum">(ArrayView1D<double, Stride1D.Dense>) Buffer for the sum</param>
            ///<param name="dim1">(int) First dimension of aView</param>
            ///<param name="dim2">(int) Second dimension of aView</param>
            for (int i = 0; i < dim1; i++)
            {

                sum[index] += aView[new Index1D(i)];

            }
            //Atomic.Add(ref sum[new Index1D(0)], aView[index]); 
        }
        static void NewtonStepKernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> uView,
            ArrayView1D<double, Stride1D.Dense> wView,
            ArrayView1D<double, Stride1D.Dense> q1View,
            ArrayView1D<double, Stride1D.Dense> q2View,
            ArrayView1D<double, Stride1D.Dense> d1View,
            ArrayView1D<double, Stride1D.Dense> d2View,
            double t)
        {
            double q1i = 1.0 / (uView[index] + wView[index]);
            double q2i = 1.0 / (uView[index] - wView[index]);
            q1View[index] = q1i;
            q2View[index] = q2i;
            d1View[index] = ((q1i * q1i) + (q2i * q2i)) / t;
            d2View[index] = ((q1i * q1i) - (q2i * q2i)) / t;

        }

        static void DualityGapKernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> NuView,
            ArrayView1D<double, Stride1D.Dense> MaxXnu,
            double lambda,
            int NuLength)
        {
            double mxxnu = MaxXnu[new Index1D(0)];

            if (mxxnu > lambda)
            {
                double lnu = lambda / mxxnu;
                
                NuView[index] = NuView[index] * lnu;
            }

        }
        static void DotKernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> aView,
            ArrayView1D<double, Stride1D.Dense> bView,
            ArrayView1D<double, Stride1D.Dense> Output,
            int length
            )
        {
            for (int i = 0; i < length; i++)
            {
                Output[index] += aView[new Index1D(i)] * bView[new Index1D(i)];
            }

        }
        static void DotSelfKernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> aView,
            ArrayView1D<double, Stride1D.Dense> Output,
            int length
            )
        {
            for (int i = 0; i < length; i++)
            {
                Output[index] += aView[new Index1D(i)] * aView[new Index1D(i)];
            }

        }
        static void sqrtkernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> aView)
        {
            aView[index] = Math.Sqrt(aView[index]);
        }
        static void NormKernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> aView,
            ArrayView1D<double, Stride1D.Dense> output,
            int length
            )
        {
            for (int i = 0; i < length; i++)
            {

                output[index] += Math.Abs(aView[new Index1D(i)]);
            }

        }
        static void Norm2Kernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> aView,
            ArrayView1D<double, Stride1D.Dense> output,
            int length
            )
        {
            double sum = 0.0;
            for (int i = 0; i < length; i++)
            {
                sum += aView[new Index1D(i)] * aView[new Index1D(i)];
            }
            output[index] = Math.Sqrt(sum);

        }
        static double Norm2NOGPU(double[] arr)
        {
            double sum = 0.0;
            for (int i = 0; i < arr.GetLength(0); i++)
            {
                sum += arr[i] * arr[i];
            }
            return Math.Sqrt(sum);

        }
        static void PobjKernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> dot,
            ArrayView1D<double, Stride1D.Dense> norm1,
            double l)
        {
            norm1[index] = dot[index] + (l * norm1[index]);

        }
        static void DobjKernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> dot1,
            ArrayView1D<double, Stride1D.Dense> dot2,
            double dobj)
        {
            dot1[index] = Math.Max((-0.25 * dot1[index]) - dot2[index], dobj);

        }
        static void InitializeNuKernal(
            Index1D index,
            ArrayView1D<double, Stride1D.Dense> YView,
            ArrayView1D<double, Stride1D.Dense> zView,
            ArrayView1D<double, Stride1D.Dense> NuView)
        {
            zView[index] = zView[index] - YView[index];
            NuView[index] = 2.0 * zView[index];

        }

        static void GetMaxValKernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> aView,
            ArrayView1D<double, Stride1D.Dense> MaxVal,
            int aViewLength)
        {
            MaxVal[index] = Math.Abs(aView[new Index1D(0)]);
            for (int i = 0; i < aViewLength; i++)
            {
                if (Math.Abs(aView[new Index1D(i)]) > MaxVal[index])
                {
                    MaxVal[index] = Math.Abs(aView[new Index1D(i)]);
                }
            }


        }
        static void FFillKernal(
            Index1D index,
            ArrayView1D<double, Stride1D.Dense> uView,
            ArrayView1D<double, Stride1D.Dense> wView,
            ArrayView2D<double, Stride2D.DenseX> fView)
        {

            Index2D ind1 = new Index2D(0, index.X);
            Index2D ind2 = new Index2D(1, index.X);
            fView[ind1] = wView[index] - uView[index];
            fView[ind2] = (-1.0 * wView[index]) - uView[index];

        }
        static void columnMeansKernal(
            Index1D index,
            ArrayView2D<double, Stride2D.DenseX> aView,
            ArrayView1D<double, Stride1D.Dense> bView,
            int y
            )
        {
            double sum = 0.0;
            for (int i = 0; i < y; i++)
            {
                sum += aView[new Index2D(i, index.X)];
            }
            bView[index] = sum / y;
        }
        static void columnMeans1DKernal(
            Index1D index,
            ArrayView1D<double, Stride1D.Dense> aView,
            ArrayView1D<double, Stride1D.Dense> bView,
            int y
            )
        {
            double sum = 0.0;
            for (int i = 0; i < y; i++)
            {
                sum += aView[new Index1D(i)];
            }
            bView[index] = sum / y;
        }
        static void subByColumnsKernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> aView,
            ArrayView1D<double, Stride1D.Dense> bView,
            ArrayView1D<double, Stride1D.Dense> meanView)
        {
            //Index1D ind2d;
            // for (int i = 0; i < n; i++)
            // {
             //   ind2d = new Index1D(i);
            bView[index] = aView[index] - meanView[new Index1D(0)];
            //}
        }
        static void columnSTDevKernal(
            Index1D index,
            ArrayView2D<double, Stride2D.DenseX> aView,
            ArrayView1D<double, Stride1D.Dense> bView,
            ArrayView1D<double, Stride1D.Dense> meanView,
            int y
            )
        {
            double sum = 0.0;
            double val;
            for (int i = 0; i < y; i++)
            {
                val = meanView[index] - aView[new Index2D(i, index.X)];
                sum += val * val;
            }
            bView[index] = Math.Sqrt(sum /y);
        }
        static void fillX2Kernal(
            Index1D index,
            ArrayView2D<double, Stride2D.DenseX> aView,
            ArrayView2D<double, Stride2D.DenseX> bView,
            ArrayView1D<double, Stride1D.Dense> colMeans,
            ArrayView1D<double, Stride1D.Dense> colSTDs,
            double c,
            double padding,
            int n,
            int m
            )
        {
            for (int i = 0; i < m; i++)
            {
                bView[new Index2D(i, index.X)] = c * (aView[new Index2D(i, index.X)] - colMeans[index]) / colSTDs[index];

            }

            bView[new Index2D(index.X + m, index.X)] = padding;
        }
        static void subIntoKernal(
            Index2D index,
            ArrayView2D<double, Stride2D.DenseX> aView,
            ArrayView2D<double, Stride2D.DenseX> bView,
            double subvalue
            )
        {
            bView[index] = aView[index] - subvalue;
        }
        static void MatrixMultiplyAcceleratedKernel(
            Index1D index,
            ArrayView2D<double, Stride2D.DenseX> aView,
            ArrayView1D<double, Stride1D.Dense> bView,
            ArrayView1D<double, Stride1D.Dense> cView)
        {
            ///<summary> Does Matrix Multiplication on two arrayviews, and then stores in a new arrayview </summary>
            ///<param name="index">(Index2D) Index to iterate through the ArrayView</param>
            ///<param name="aView">(ArrayView2D<double, Stride2D.DenseX>) 1st ArrayView being multiplied</param>
            ///<param name="bView">(ArrayView2D<double, Stride2D.DenseX>) 2nd ArrayView being multiplied</param>
            ///<param name="cView">(ArrayView2D<double, Stride2D.DenseX>) Buffer where new value goes</param>
            var x = index.X;
            //var y = index.Y;
            double sum = 0.0;
            for (var i = 0; i < aView.Extent.ToIntIndex().Y; i++){
                sum += aView[new Index2D(x, i)] * bView[new Index1D(i)];
            }

            cView[index] = sum;
        }
        static void MatrixMultiply2DKernel(
            Index2D index,
            ArrayView2D<double, Stride2D.DenseX> aView,
            ArrayView2D<double, Stride2D.DenseX> bView,
            ArrayView2D<double, Stride2D.DenseX> cView)
        {
            ///<summary> Does Matrix Multiplication on two arrayviews, and then stores in a new arrayview </summary>
            ///<param name="index">(Index2D) Index to iterate through the ArrayView</param>
            ///<param name="aView">(ArrayView2D<double, Stride2D.DenseX>) 1st ArrayView being multiplied</param>
            ///<param name="bView">(ArrayView2D<double, Stride2D.DenseX>) 2nd ArrayView being multiplied</param>
            ///<param name="cView">(ArrayView2D<double, Stride2D.DenseX>) Buffer where new value goes</param>
            var x = index.X;
            var y = index.Y;
            double sum = 0.0;
            for (var i = 0; i < aView.IntExtent.Y; i++)
                sum += aView[new Index2D(x, i)] * bView[new Index2D(i, y)];

            cView[index] = sum;
        }
        static void SMMatMul2DKernal(Index3D index,
            ArrayView2D<double, Stride2D.DenseX> aView,
            ArrayView2D<double, Stride2D.DenseX> bView,
            ArrayView2D<double, Stride2D.DenseX> cView){

            var x = index.X;
            var y = index.Y;
            var z = index.Z;
            double val = aView[new Index2D(x, y)] * aView[new Index2D(z, y)];
            Atomic.Add(ref cView[new Index2D(x, z)], val);


        }
        static void SMMatMul1DKernal(Index2D index,
            ArrayView2D<double, Stride2D.DenseX> aView,
            ArrayView1D<double, Stride1D.Dense> bView,
            ArrayView1D<double, Stride1D.Dense> cView){

            var x = index.X;
            var y = index.Y;
            
            
            double val = aView[new Index2D(x,y)] * bView[new Index1D(y)];
            //Group.Barrier();
            Atomic.Add(ref cView[new Index1D(x)], val);


        }
        static void SMMatMul1DKernalTooLarge(Index2D index,
            ArrayView2D<double, Stride2D.DenseX> aView,
            ArrayView1D<double, Stride1D.Dense> bView,
            ArrayView1D<double, Stride1D.Dense> cView){

            var x = index.X;
            var y = index.Y;
            
            
            double val = aView[new Index2D(y,x)] * bView[new Index1D(x)];
            //Group.Barrier();
            Atomic.Add(ref cView[new Index1D(y)], val);


        }
        static void MatrixMultiplyGradphiKernel(
            Index1D index,
            ArrayView2D<double, Stride2D.DenseX> aView,
            ArrayView1D<double, Stride1D.Dense> bView,
            ArrayView2D<double, Stride2D.DenseX> cView)
        {
            ///<summary> Does Matrix Multiplication on two arrayviews, and then stores in a new arrayview </summary>
            ///<param name="index">(Index2D) Index to iterate through the ArrayView</param>
            ///<param name="aView">(ArrayView2D<double, Stride2D.DenseX>) 1st ArrayView being multiplied</param>
            ///<param name="bView">(ArrayView2D<double, Stride2D.DenseX>) 2nd ArrayView being multiplied</param>
            ///<param name="cView">(ArrayView2D<double, Stride2D.DenseX>) Buffer where new value goes</param>
            int x = index.X;
            //var y = index.Y;
            double sum = 0.0;
            for (int i = 0; i < aView.IntExtent.Y; i++)
                sum += aView[new Index2D(x, i)] * bView[new Index1D(i)];

            cView[new Index2D(0, x)] = sum;
        }
        static void setBuffToValueKernal(Index1D index, ArrayView1D<double, Stride1D.Dense> buff, double setvalue)
        {
            buff[index] = setvalue;
        }
        void print2d(double[,] array)
        {
            Console.WriteLine(array);

            for (int i = 0; i < array.GetLength(0); i++)
            {
                Console.Write("[");
                for (int j = 0; j < array.GetLength(1); j++)
                {
                    Console.Write("{0}, ", array[i, j]);
                }
                Console.Write("]");
                Console.WriteLine(", ");
            }
            Console.WriteLine("]");
        }
        void print3d(double[,,] array)
        {
            for (int i = 0; i < array.GetLength(0); i++)
            {
                Console.Write("[");
                for (int j = 0; j < array.GetLength(1); j++)
                {
                    Console.Write("[");
                    for (int k = 0; k < array.GetLength(2); k++)
                    {
                        Console.Write("{0}, ", array[i, j, k]);
                    }
                    Console.Write("]");
                    Console.WriteLine(", ");
                }
                Console.Write("]");
                Console.WriteLine(", ");
                Console.WriteLine();
            }
            Console.WriteLine("]");

        }
        void print1d(double[] array)
        {
            Console.Write("[");
            for (int j = 0; j < array.GetLength(0); j++)
            {
                Console.Write("{0}, ", array[j]);
            }
            Console.WriteLine("]");

        }
        public double[] predict(double[,] x)
            ///<summary>Predicts output based off of x</summary>
            ///<param name="x">Array of inputs</param>
        { 
            //return applyadd(matrixmul(x, this.W), this.B);
            return applyadd(matrixmul(x, this.W), this.B);
        }
        double[] applyadd(double[] arr, double val)
        ///<summary>Adds a value to each member of a 2d array</summary>
        {
            double[] temp = new double[arr.GetLength(0)];
            for (int i = 0; i < arr.GetLength(0); i++)
            {
            
                temp[i] = arr[i] + val;
                
            }
            return temp;

        }
        double[] matrixmul(double[,] x, double[] y)
        {
            ///<summary>Does matrix multiplication on two 2d arrays</summary>
            ///<param name="x">Array 1</param>
            ///<param name="y">Array 2</param>


            //Initialize all varaibles
            int m = x.GetLength(0), n = x.GetLength(1), p = y.GetLength(0);

            /////Create empty array of new size
            double[] c = new double[m];

            //double sum = 0.0;
            for(int i = 0; i < m; i++){
                //sum = 0.0;
                for(int j = 0; j< n; j++){
                    c[i] += x[i,j] * y[j]; 
                }
            }
            return c;
        }
    // }




    // public class ElasticRegression
    // {
    //     int Iterations;
    //     int M;
    //     int N;
    //     int Q;

    //     double Learning_rate;
    //     double L1_penality;
    //     double L2_penality;
    //     double B;

    //     double[,] W;
    //     double[,] X;
    //     double[,] Y;

    //     //Memory Buffers Used in GPU computations
    //     MemoryBuffer2D<double, Stride2D.DenseX> XBuffer;
    //     MemoryBuffer2D<double, Stride2D.DenseX> YBuffer;
    //     MemoryBuffer2D<double, Stride2D.DenseX> WBuffer;
    //     MemoryBuffer1D<double, Stride1D.Dense> SumBuffer;
    //     MemoryBuffer2D<double, Stride2D.DenseX> MatMulBuffer;
    //     MemoryBuffer2D<double, Stride2D.DenseX> PredMatMulBuffer;
    //     MemoryBuffer2D<double, Stride2D.DenseX> dWBuffer;
    //     MemoryBuffer2D<double, Stride2D.DenseX> YDiffBuffer;
    //     MemoryBuffer2D<double, Stride2D.DenseX> YPredBuffer;

    //     //Different Kernals needed for GPU computations
    //     Action<Index2D,
    //             ArrayView2D<double, Stride2D.DenseX>,
    //             ArrayView2D<double, Stride2D.DenseX>,
    //             ArrayView2D<double, Stride2D.DenseX>> subtwoarrkern;
    //     Action<Index2D,
    //             ArrayView2D<double, Stride2D.DenseX>,
    //             ArrayView2D<double, Stride2D.DenseX>> subtwoarrkern2;
    //     Action<Index2D,
    //             ArrayView2D<double, Stride2D.DenseX>,
    //             ArrayView2D<double, Stride2D.DenseX>,
    //             ArrayView2D<double, Stride2D.DenseX>> matrixmulkern;
    //     Action<Index2D,
    //             ArrayView2D<double, Stride2D.DenseX>,
    //             ArrayView2D<double, Stride2D.DenseX>,
    //             ArrayView2D<double, Stride2D.DenseX>,
    //             double, double, double> updatekernel;
    //     Action<Index2D,
    //             ArrayView2D<double, Stride2D.DenseX>,
    //             double> matrixaddkern;
    //     Action<Index2D,
    //             ArrayView2D<double, Stride2D.DenseX>,
    //             double> matrixmulsinglekern;
    //     Action<Index1D,
    //             ArrayView2D<double, Stride2D.DenseX>,
    //             ArrayView1D<double, Stride1D.Dense>,
    //             int, int> sumofkern;
    //     Action<Index1D,
    //             ArrayView1D<double, Stride1D.Dense>> clearsumkern;




    //     Device dev;
    //     Accelerator accelerate;

    //     //Will be used later when optimized for GPU use
    //     Context context;


    //     /////      Constructor
    //     public ElasticRegression(double learning_rate, int iterations, double l1_penality, double l2_penality, bool fullGPU = false)
    //     {
    //         ///<summary>Constructor for ElasticRegression object</summary>
    //         ///<param name="learning_rate">(double) learning rate of the regression</param>
    //         ///<param name="iterations">(int) How many iterations of the algorithm will run when the model is fit</param>
    //         ///<param name="l1_penality">(double )L1 penality</param>
    //         ///<param name="l2_penality">(double )L2 penality</param>




    //         this.Learning_rate = learning_rate;

    //         this.Iterations = iterations;

    //         this.L1_penality = l1_penality;

    //         this.L2_penality = l2_penality;
    //         this.context = Context.Create(builder => builder.Default());//builder => builder.Default().EnableAlgorithms())
    //         this.dev = this.context.GetPreferredDevice(preferCPU: false);

    //     }

    //     public ElasticRegression fitFULLGPU(double[,] X, double[,] Y, bool verbose = true)
    //     {
    //         ///<summary>Trains the model</summary>
    //         ///<param name="X">(double[,]) A 2d array of the inputs to be trained on.</param>
    //         ///<param name="Y">(double[,]) A 2d array of the target outputs, must have same length as X</param>
    //         ///<param name="verbose">(boolean) Determines if the program outputs updates as it runs, default = true</param>


    //         //Number of training examples
    //         this.M = X.GetLength(0) * X.GetLength(1);
    //         //Number of features
    //         this.N = X.GetLength(1);

    //         //Initializes accelerator
    //         this.accelerate = this.dev.CreateAccelerator(this.context);

    //         //Initialize Kernels for use with the GPU
    //         this.subtwoarrkern = this.accelerate.LoadAutoGroupedStreamKernel<
    //             Index2D,
    //             ArrayView2D<double, Stride2D.DenseX>,
    //             ArrayView2D<double, Stride2D.DenseX>,
    //             ArrayView2D<double, Stride2D.DenseX>>(
    //             subtwoarrsKernal);
    //         this.subtwoarrkern2 = this.accelerate.LoadAutoGroupedStreamKernel<
    //             Index2D,
    //             ArrayView2D<double, Stride2D.DenseX>,
    //             ArrayView2D<double, Stride2D.DenseX>>(
    //             subtwoarrsKernal2);
    //         this.matrixmulsinglekern = this.accelerate.LoadAutoGroupedStreamKernel<
    //             Index2D,
    //             ArrayView2D<double, Stride2D.DenseX>,
    //             double>(
    //             multKernal);
    //         this.matrixaddkern = this.accelerate.LoadAutoGroupedStreamKernel<
    //             Index2D,
    //             ArrayView2D<double, Stride2D.DenseX>,
    //             double>(
    //             additionKernal);
    //         this.matrixmulkern = this.accelerate.LoadAutoGroupedStreamKernel<
    //             Index2D,
    //             ArrayView2D<double, Stride2D.DenseX>,
    //             ArrayView2D<double, Stride2D.DenseX>,
    //             ArrayView2D<double, Stride2D.DenseX>>(
    //             MatrixMultiplyAcceleratedKernel);
    //         this.updatekernel = this.accelerate.LoadAutoGroupedStreamKernel<
    //             Index2D,
    //             ArrayView2D<double, Stride2D.DenseX>,
    //             ArrayView2D<double, Stride2D.DenseX>,
    //             ArrayView2D<double, Stride2D.DenseX>,
    //             double, double, double>(
    //             updateweightskernal);
    //         this.sumofkern = this.accelerate.LoadAutoGroupedStreamKernel<
    //             Index1D,
    //             ArrayView2D<double, Stride2D.DenseX>,
    //             ArrayView1D<double, Stride1D.Dense>,
    //             int, int>(sumofkernal);
    //         this.clearsumkern = this.accelerate.LoadAutoGroupedStreamKernel<
    //             Index1D,
    //             ArrayView1D<double, Stride1D.Dense>>(clearsumbuff);


    //         //Number of outputs
    //         this.Q = Y.GetLength(1);

    //         //Initializes variables
    //         //The weights of the model
    //         this.W = new double[this.N, this.Q];

    //         //Buffer to store the weights during GPU running
    //         this.WBuffer = this.accelerate.Allocate2DDenseX<double>(new Index2D(this.N, this.Q));

    //         //Buffer to store the temporary weight changes
    //         this.dWBuffer = this.accelerate.Allocate2DDenseX<double>(new Index2D(this.N, this.Q));


    //         this.B = 0.0;


    //         //Initializes the buffer for training data features
    //         this.X = X;
    //         this.XBuffer = this.accelerate.Allocate2DDenseX<double>(new Index2D(X.GetLength(0), X.GetLength(1)));
    //         XBuffer.CopyFromCPU(this.X);

    //         //Initializes the buffer for training data outputs
    //         this.Y = Y;
    //         this.YBuffer = this.accelerate.Allocate2DDenseX<double>(new Index2D(Y.GetLength(0), Y.GetLength(1)));
    //         YBuffer.CopyFromCPU(this.Y);


    //         //Buffers for working with prediction and diff of Y
    //         this.YPredBuffer = this.accelerate.Allocate2DDenseX<double>(new Index2D(Y.GetLength(0), Y.GetLength(1)));
    //         this.YDiffBuffer = this.accelerate.Allocate2DDenseX<double>(new Index2D(Y.GetLength(1), Y.GetLength(0)));

    //         //Buffers used for matrix multiplication
    //         this.MatMulBuffer = accelerate.Allocate2DDenseX<double>(new Index2D(Y.GetLength(1), X.GetLength(1)));
    //         this.PredMatMulBuffer = accelerate.Allocate2DDenseX<double>(new Index2D(X.GetLength(0), this.Q));

    //         //Helper buffer to calculate the sum of the YDiffBuffer
    //         this.SumBuffer = accelerate.Allocate1D<double>(1L);

    //         double db = 0.0;

    //         //Gradient descent learning


    //         //Using Buffers in order to edit them on each iteration
    //         using (this.YBuffer)
    //         using (this.YPredBuffer)
    //         using (this.YDiffBuffer)
    //         using (this.MatMulBuffer)
    //         using (this.XBuffer)
    //         using (this.WBuffer)
    //         using (this.dWBuffer)
    //         using (this.SumBuffer)
    //         using (this.accelerate)
    //         {
    //             for (int i = 0; i < this.Iterations; i++)
    //             {
    //                 if (verbose)
    //                 {
    //                     Console.WriteLine("Iteration {0}/{1}", i, this.Iterations);
    //                 }

    //                 //Gets the prediction based on the WBuffer
    //                 this.matrixmulkern(this.YPredBuffer.Extent.ToIntIndex(), this.XBuffer, this.WBuffer, this.YPredBuffer);
    //                 this.matrixaddkern(this.YPredBuffer.Extent.ToIntIndex(), this.YPredBuffer, this.B);


    //                 //Gets the difference between YActual and YPrediction and puts it in YDiffBuffer
    //                 this.subtwoarrkern(this.YBuffer.Extent.ToIntIndex(), this.YBuffer, this.YPredBuffer, this.YDiffBuffer);

    //                 //Multiply error by the actual xbuffer
    //                 this.matrixmulkern(this.MatMulBuffer.Extent.ToIntIndex(), this.YDiffBuffer, this.XBuffer, this.MatMulBuffer);


    //                 //Update dWbuffer
    //                 this.updatekernel(this.WBuffer.Extent.ToIntIndex(), this.WBuffer.View, this.MatMulBuffer.View, this.dWBuffer.View, this.L1_penality, this.L2_penality, this.M);

    //                 //Set Sumbuffer to 0
    //                 this.clearsumkern(this.SumBuffer.Extent.ToIntIndex(), this.SumBuffer.View);

    //                 //Get the sum of YDiff Buffer
    //                 this.sumofkern(this.SumBuffer.Extent.ToIntIndex(), this.YDiffBuffer.View, this.SumBuffer.View, this.YDiffBuffer.Extent.ToIntIndex().X, this.YDiffBuffer.Extent.ToIntIndex().Y);


    //                 db = (-2.0 * this.SumBuffer.GetAsArray1D()[0]) / this.M;

    //                 //Multiply dWBuffer by the learning rate
    //                 this.matrixmulsinglekern(this.dWBuffer.Extent.ToIntIndex(), this.dWBuffer, this.Learning_rate);

    //                 //Sub the dWBuffer from the WBuffer
    //                 this.subtwoarrkern2(this.WBuffer.Extent.ToIntIndex(), this.WBuffer, this.dWBuffer);

    //                 this.B = this.B - (this.Learning_rate * db);

    //             }
    //             //Move the weights out of GPU to be used for predictions
    //             this.W = this.WBuffer.GetAsArray2D();
    //         }

    //         this.accelerate.Dispose();
    //         return this;
    //     }

    //     static void clearsumbuff(Index1D index, ArrayView1D<double, Stride1D.Dense> sum)
    //     {
    //         ///<summary>Sets the sumbuffer to 0</summary>
    //         ///<param name="index">(Index1D) Index to iterate through the array</param>
    //         ///<param name="sum">(ArrayView1D<double, Stride1D.Dense>) Buffer</param>
    //         sum[index] = 0.0;
    //     }
    //     static void sumofkernal(Index1D index, ArrayView2D<double, Stride2D.DenseX> aView, ArrayView1D<double, Stride1D.Dense> sum, int dim1, int dim2)
    //     {
    //         ///<summary>Adds up everything in a 2DBuffer.</summary>
    //         ///<param name="index">(Index1D) Index to iterate through the array</param>
    //         ///<param name="aView">(ArrayView2D<double, Stride2D.DenseX>) Buffer to be added up</param>
    //         ///<param name="sum">(ArrayView1D<double, Stride1D.Dense>) Buffer for the sum</param>
    //         ///<param name="dim1">(int) First dimension of aView</param>
    //         ///<param name="dim2">(int) Second dimension of aView</param>
    //         for (int i = 0; i < dim1; i++)
    //         {
    //             for (int j = 0; j < dim2; j++)
    //             {
    //                 sum[index] += aView[new Index2D(i, j)];
    //             }
    //         }
    //     }


    //     static double[,] kernalTester(Accelerator accelerator, double[,] a, double[,] b)
    //     {
    //         var m = a.GetLength(0);
    //         var ka = a.GetLength(1);
    //         var kb = b.GetLength(0);
    //         var n = b.GetLength(1);

    //         var kernal = accelerator.LoadAutoGroupedStreamKernel<
    //             Index2D,
    //             ArrayView2D<double, Stride2D.DenseX>,
    //             ArrayView2D<double, Stride2D.DenseX>,
    //             ArrayView2D<double, Stride2D.DenseX>>(
    //             subtwoarrsKernal);

    //         using var aBuffer = accelerator.Allocate2DDenseX<double>(new Index2D(m, ka));
    //         using var bBuffer = accelerator.Allocate2DDenseX<double>(new Index2D(m, ka));
    //         using var cBuffer = accelerator.Allocate2DDenseX<double>(new Index2D(m, ka));

    //         aBuffer.CopyFromCPU(a);
    //         bBuffer.CopyFromCPU(b);

    //         kernal(aBuffer.Extent.ToIntIndex(), aBuffer.View, bBuffer.View, cBuffer.View);

    //         return cBuffer.GetAsArray2D();
    //     }

    //     static void updateweightskernal(Index2D index,
    //         ArrayView2D<double, Stride2D.DenseX> WView,
    //         ArrayView2D<double, Stride2D.DenseX> MMView,
    //         ArrayView2D<double, Stride2D.DenseX> DwView,
    //         double L1,
    //         double L2,
    //         double M)
    //     {
    //         ///<summary> Update dWBuffer</summary>
    //         ///<param name="index">(Index2D) Index to iterate through the ArrayView</param>
    //         ///<param name="WView">(ArrayView2D<double, Stride2D.DenseX>) Weight buffer </param>
    //         ///<param name="MMView">(ArrayView2D<double, Stride2D.DenseX>) Matrix Mul Buffer</param>
    //         ///<param name="DwView">(ArrayView2D<double, Stride2D.DenseX>) New weight buffer</param>
    //         ///<param name="L1">(double) L1 penalty</param>
    //         ///<param name="L2">(double) L2 penalty</param>
    //         ///<param name="M">(double) Total size of data</param>

    //         if (WView[index] > 0)
    //         {
    //             DwView[index] = (((-1 * MMView[new Index2D(index.Y, index.X)] * (2.0 + L1))) + (L2 * WView[index])) / M; //(((-this.multipliedMatrix[z, j] * (2.0 + this.L1_penality))) + (2 * this.L2_penality * this.W[j, z])) / this.M; ;
    //         }
    //         else
    //         {
    //             DwView[index] = (((-1 * MMView[new Index2D(index.Y, index.X)] * (2.0 - L1))) + (L2 * WView[index])) / M; //(((-this.multipliedMatrix[z, j] * (2.0 - this.L1_penality))) + (2 * this.L2_penality * this.W[j, z])) / this.M; ; 
    //         }

    //     }
    //     static void additionKernal(
    //         Index2D index,
    //         ArrayView2D<double, Stride2D.DenseX> aView,
    //         double addvalue
    //         )
    //     {
    //         ///<summary> Adds a single value to every element in an ArrayView </summary>
    //         ///<param name="index">(Index2D) Index to iterate through the ArrayView</param>
    //         ///<param name="aView">(ArrayView2D<double, Stride2D.DenseX>) Buffer the value is being added to</param>
    //         ///<param name="addvalue">(double) Value being added to the ArrayView</param>
    //         aView[index] += addvalue;
    //     }
    //     static void multKernal(
    //         Index2D index,
    //         ArrayView2D<double, Stride2D.DenseX> aView,
    //         double multvalue
    //         )
    //     {
    //         ///<summary> Adds a single value to every element in an ArrayView </summary>
    //         ///<param name="index">(Index2D) Index to iterate through the ArrayView</param>
    //         ///<param name="aView">(ArrayView2D<double, Stride2D.DenseX>) Buffer the value is being added to</param>
    //         ///<param name="multvalue">(double) Value being multiplied to every element of the ArrayView</param>
    //         aView[index] = aView[index] * multvalue;
    //     }

    //     static void subKernal(
    //         Index2D index,
    //         ArrayView2D<double, Stride2D.DenseX> aView,
    //         double subvalue
    //         )
    //     {
    //         ///<summary> Adds a single value to every element in an ArrayView </summary>
    //         ///<param name="index">(Index2D) Index to iterate through the ArrayView</param>
    //         ///<param name="aView">(ArrayView2D<double, Stride2D.DenseX>) Buffer the value is being added to</param>
    //         ///<param name="subvalue">(double) Value being subtracted to the ArrayView</param>
    //         aView[index] = aView[index] - subvalue;
    //     }
    //     static void divKernal(
    //         Index2D index,
    //         ArrayView2D<double, Stride2D.DenseX> aView,
    //         double divvalue
    //         )
    //     {
    //         ///<summary> Adds a single value to every element in an ArrayView </summary>
    //         ///<param name="index">(Index2D) Index to iterate through the ArrayView</param>
    //         ///<param name="aView">(ArrayView2D<double, Stride2D.DenseX>) Buffer the value is being added to</param>
    //         ///<param name="divvalue">(double) Value being divided from the ArrayView</param>
    //         aView[index] = aView[index] / divvalue;


    //     }

    //     static void subtwoarrsKernal(
    //         Index2D index,
    //         ArrayView2D<double, Stride2D.DenseX> aView,
    //         ArrayView2D<double, Stride2D.DenseX> bView,
    //         ArrayView2D<double, Stride2D.DenseX> cView)
    //     {
    //         ///<summary> Subtracts one ArrayView from another, and puts it in a new ArrayView with dimensions reversed (Used for setting up YDiffBuffer) </summary>
    //         ///<param name="index">(Index2D) Index to iterate through the ArrayView</param>
    //         ///<param name="aView">(ArrayView2D<double, Stride2D.DenseX>) Buffer being subtracted from</param>
    //         ///<param name="bView">(ArrayView2D<double, Stride2D.DenseX>) Buffer being subtracted</param>
    //         ///<param name="cView">(ArrayView2D<double, Stride2D.DenseX>) Buffer where new value goes</param>

    //         cView[new Index2D(index.Y, index.X)] = aView[index] - bView[index];

    //     }
    //     static void subtwoarrsKernal2(
    //         Index2D index,
    //         ArrayView2D<double, Stride2D.DenseX> aView,
    //         ArrayView2D<double, Stride2D.DenseX> bView)
    //     {
    //         ///<summary> Subtracts one ArrayView from another </summary>
    //         ///<param name="index">(Index2D) Index to iterate through the ArrayView</param>
    //         ///<param name="aView">(ArrayView2D<double, Stride2D.DenseX>) Buffer being subtracted from</param>
    //         ///<param name="bView">(ArrayView2D<double, Stride2D.DenseX>) Buffer being subtracted</param>

    //         aView[index] = aView[index] - bView[index];

    //     }

    //     static void MatrixMultiplyAcceleratedKernel(
    //         Index2D index,
    //         ArrayView2D<double, Stride2D.DenseX> aView,
    //         ArrayView2D<double, Stride2D.DenseX> bView,
    //         ArrayView2D<double, Stride2D.DenseX> cView)
    //     {
    //         ///<summary> Does Matrix Multiplication on two arrayviews, and then stores in a new arrayview </summary>
    //         ///<param name="index">(Index2D) Index to iterate through the ArrayView</param>
    //         ///<param name="aView">(ArrayView2D<double, Stride2D.DenseX>) 1st ArrayView being multiplied</param>
    //         ///<param name="bView">(ArrayView2D<double, Stride2D.DenseX>) 2nd ArrayView being multiplied</param>
    //         ///<param name="cView">(ArrayView2D<double, Stride2D.DenseX>) Buffer where new value goes</param>
    //         var x = index.X;
    //         var y = index.Y;
    //         var sum = 0.0;
    //         for (var i = 0; i < aView.IntExtent.Y; i++)
    //             sum += aView[new Index2D(x, i)] * bView[new Index2D(i, y)];

    //         cView[index] = sum;
    //     }

    //     //Predicts outputs based off of x
    //     double[,] predict(double[,] x)
    //     ///<summary>Predicts output based off of x</summary>
    //     ///<param name="x">Array of inputs</param>
    //     {
    //         this.accelerate = this.dev.CreateAccelerator(this.context);
    //         double[,] prediction = applyadd(MatrixMultiplyAccelerated(this.accelerate, x, this.W), this.B);
    //         this.accelerate.Dispose();
    //         return prediction;
    //     }


    //     //Used in prediction to move the arrays onto the GPU to set up to be multiplied
    //     static double[,] MatrixMultiplyAccelerated(Accelerator accelerator, double[,] a, double[,] b)
    //     {
    //         var m = a.GetLength(0);
    //         var ka = a.GetLength(1);
    //         var kb = b.GetLength(0);
    //         var n = b.GetLength(1);

    //         if (ka != kb)
    //             throw new ArgumentException($"Cannot multiply {m}x{ka} matrix by {n}x{kb} matrix", nameof(b));

    //         var kernel = accelerator.LoadAutoGroupedStreamKernel<
    //             Index2D,
    //             ArrayView2D<double, Stride2D.DenseX>,
    //             ArrayView2D<double, Stride2D.DenseX>,
    //             ArrayView2D<double, Stride2D.DenseX>>(
    //             MatrixMultiplyAcceleratedKernel);

    //         using var aBuffer = accelerator.Allocate2DDenseX<double>(new Index2D(m, ka));
    //         using var bBuffer = accelerator.Allocate2DDenseX<double>(new Index2D(ka, n));
    //         using var cBuffer = accelerator.Allocate2DDenseX<double>(new Index2D(m, n));
    //         aBuffer.CopyFromCPU(a);
    //         bBuffer.CopyFromCPU(b);

    //         kernel(cBuffer.Extent.ToIntIndex(), aBuffer.View, bBuffer.View, cBuffer.View);
    //         return cBuffer.GetAsArray2D();
    //     }
    //     //Adds a value to each member of a 2d array
    //     double[,] applyadd(double[,] arr, double val)
    //     ///<summary>Adds a value to each member of a 2d array</summary>
    //     {
    //         double[,] temp = new double[arr.GetLength(0), arr.GetLength(1)];
    //         for (int i = 0; i < arr.GetLength(0); i++)
    //         {
    //             for (int j = 0; j < arr.GetLength(1); j++)
    //             {
    //                 temp[i, j] = arr[i, j] + val;
    //             }
    //         }
    //         return temp;

    //     }

    //     /*
    //     Lines 494-709 are all methods used for testing the algo, will be deleted on final version
    //     */
    //     //Helper function used for testing, prints 1d array
    //     void print1d(double[] array)
    //     {

    //         for (int j = 0; j < array.GetLength(0); j++)
    //         {
    //             Console.Write("{0} ", array[j]);
    //         }

    //     }
    //     //Helper function used for testing, prints 2d array
    //     void print2d(double[,] array)
    //     {
    //         Console.Write("[");
    //         for (int i = 0; i < array.GetLength(0); i++)
    //         {
    //             Console.Write("[");
    //             for (int j = 0; j < array.GetLength(1); j++)
    //             {
    //                 Console.Write("{0}, ", array[i, j]);
    //             }
    //             Console.Write("]");
    //             Console.Write(", ");
    //         }
    //         Console.WriteLine("]");
    //     }


        void writetoCSV(double[,] array, string path, string inorout)
        {
            StreamWriter file = new StreamWriter(path);
            var iLength = array.GetLength(0);
            var jLength = array.GetLength(1);
            for (int k = 0; k < jLength; k++)
            {

                if (k == jLength - 1)
                {
                    file.Write("{1}{0}", k, inorout);
                }
                else
                {
                    file.Write("{1}{0},", k, inorout);
                }

            }
            file.WriteLine();
            for (int j = 0; j < iLength; j++)
            {

                for (int i = 0; i < jLength; i++)
                {
                    if (i == jLength - 1)
                    {
                        file.Write("{0}", array[j, i]);
                    }
                    else
                    {
                        file.Write("{0},", array[j, i]);
                    }

                }
                file.WriteLine();
                file.Flush();
            }
        }

        void writetoCSVFullClean(double[,] array1, double[] array2, string path)
        {
            StreamWriter file = new StreamWriter(path);
            var iLength = array1.GetLength(0);
            var jLength = array1.GetLength(1);
            //var kLength = array2.GetLength(1);
            for (int k = 0; k < jLength; k++)
            {

                if (k == jLength - 1)
                {
                    file.Write("{1}{0},", k, "IN");
                }
                else
                {
                    file.Write("{1}{0},", k, "IN");
                }

            }
            
            
            file.Write("{1}{0}", 1, "OUT");
            
           
            file.WriteLine();
            file.Flush();
            for (int j = 0; j < iLength; j++)
            {

                for (int i = 0; i < jLength; i++)
                {
                    if (i == jLength - 1)
                    {
                        file.Write("{0},", array1[j, i]);
                    }
                    else
                    {
                        file.Write("{0},", array1[j, i]);
                    }

                }
                
                
                file.Write("{0}", array2[j]);
                
                
                //     if (z == kLength - 1)
                //     {
                //         file.Write("{0}", array2[j, z]);
                //     }
                //     else
                //     {
                //         file.Write("{0},", array2[j, z]);
                //     }

                // }
                file.WriteLine();
                file.Flush();
            }
        }

        static bool isEqualArrs(double[,] arr1, double[,] arr2)
        {
            int counter = 0;
            if (arr1.GetLength(0) == arr2.GetLength(0) && arr1.GetLength(1) == arr2.GetLength(1))
            {
                for (int i = 0; i < arr1.GetLength(0); i++)
                {
                    for (int j = 0; j < arr1.GetLength(1); j++)
                    {
                        if (Math.Abs(arr1[i, j] - arr2[i, j]) > .1)
                        {
                            Console.WriteLine(Math.Abs(arr1[i, j] - arr2[i, j]) > .1);
                            Console.Write("Counter");
                            Console.WriteLine(i);
                            Console.WriteLine(j);
                            Console.WriteLine(counter);
                            Console.Write(arr1[i, j]);
                            Console.Write("  ");
                            Console.WriteLine(arr2[i, j]);
                            return false;
                        }
                        else
                        {
                            counter += 1;
                        }
                        //Xactual[i, 1] = ((double)q.NextDouble() * 10) + 2;

                    }
                }
                return true;

            }
            else
            {
                Console.WriteLine("DIFF LENGTHS");
                return false;
            }

        }
        double sumof(double[,] arr, int row)
        {
            double sum = 0.0;
            for (int i = 0; i < arr.GetLength(1); i++)
            {
                sum += arr[row, i];
            }
            return sum;
        }
        double[] generatedatax(int n, int m)
        {
            Random q = new Random();
            ElasticNet e = new ElasticNet();
            ElasticNet enet = new ElasticNet();
            
            double[,] tempx = new double[(int)Math.Pow(10, n), (int)Math.Pow(10, m)];
            for (int x = 0; x < tempx.GetLength(0); x++)
            {
                for (int y = 0; y < tempx.GetLength(1); y++)
                {
                    tempx[x, y] = ((double)q.NextDouble()*10);
                }
                //Xactual[i, 1] = ((double)q.NextDouble() * 10) + 2;

            }
            double[] tempy = new double[(int)Math.Pow(10, n)];

            for (int x = 0; x < tempy.GetLength(0); x++)
            {
                
                tempy[x] = (e.sumof(tempx, x));

                    ////////xConsole.WriteLine(Yactual[i,j]
                
                // Yactual[i, 0] = ((Xactual[i, 0]) * 1000 + 2000);
                // Yactual[i, 1] = ((Xactual[i, 0]) * 100 + 100);
                //Yactual[i, 1] = ((Xactual[i, 0]) * 1000 + 2000);
            }
            double ysum = 0.0;
            for (int x = 0; x < tempy.GetLength(0); x++)
            {
                ysum += tempy[x];
            }
            Console.WriteLine("y sum");
            Console.WriteLine(ysum);
            String s = "datasets/DataSet" + Math.Pow(10, n).ToString() + "x" + Math.Pow(10, m).ToString() + ".csv";
            String xs = "datasets/XDataSet" + Math.Pow(10, n).ToString() + "x" + Math.Pow(10, m).ToString() + ".csv";
            String ys = "datasets/YDataSet" + Math.Pow(10, n).ToString() + "x" + Math.Pow(10, m).ToString() + ".csv";
            e.writetoCSVFullClean(tempx, tempy, s);
            //e.writetoCSV(tempx, xs, "IN");
            //e.writetoCSV(tempy, ys, "OUT");
            Console.WriteLine(Math.Pow(10, n).ToString() + "x" + Math.Pow(10, m).ToString() );
            Stopwatch stop = new Stopwatch();
            stop.Start();
            enet.fit(tempx, tempy,Math.Pow(10, m)/2, 0.01, 1e-4, 100);
            stop.Stop();

            Console.WriteLine(s + " is Done Running");
            Console.Write("Time (seconds): ");
            Console.WriteLine(((double)stop.ElapsedMilliseconds) / 1000f);
            double[] predict = enet.predict(tempx);
            double predtotal = 0.0;
            int counter = 0;
            // Console.WriteLine(predict.GetLength(0));
            // //Console.WriteLine(predict.GetLength(1));

            // Console.WriteLine(tempy.GetLength(0));
            //Console.WriteLine(tempy.GetLength(1));


            for (int i = 0; i < predict.GetLength(0); i++)
            {
                
                predtotal += Math.Abs(tempy[i] - predict[i]);
                    // resNtotal += Math.Abs(Yactual[i,j]-resN[i,j]);
                counter += 1;

            }
            
            Console.Write("Average Error: ");
            Console.WriteLine(predtotal / counter);
            Console.ReadLine();


            double[] outf = new double[4];

            outf[0] = (double)Math.Pow(10, n);
            outf[1] = (double)Math.Pow(10, m);
            outf[2] = ((double)stop.ElapsedMilliseconds) / 1000f;
            outf[3] = predtotal / counter;

            return outf;
        }
        void test(int n, int m)
        {
            ElasticNet e = new ElasticNet();
            double[,] outputs = new double[n * m, 4];
            double[] outf = new double[4];
            int counter = 0;
            for (int i = 5; i < n; i++)
            {
                for (int j = 2; j < m; j++)
                {
                    outf = e.generatedatax(i, j);
                    outputs[counter, 0] = outf[0];
                    outputs[counter, 1] = outf[1];
                    outputs[counter, 2] = outf[2];
                    outputs[counter, 3] = outf[3];
                    counter += 1;
                }
            }
            e.writetoCSV(outputs, "TimesOfDataSMENRECS.csv", "Val");
        }

        

        static void Main(string[] args)
        {

            using var streamReader = File.OpenText("spy_train.csv");
            using var csvReader = new CsvReader(streamReader, CultureInfo.CurrentCulture);
            string[] row = new string[13];
            var users = csvReader.GetRecords<double>();
            double[,] spyx = new double[5378,11];
            double[] spyy = new double[5378];
            ElasticNet e2 = new ElasticNet();
            using var streamReader2 = File.OpenText("EnergyData.csv");
            using var csvReader2 = new CsvReader(streamReader2, CultureInfo.CurrentCulture);

            using var streamReader3 = File.OpenText("CASP.csv");
            using var csvReader3 = new CsvReader(streamReader2, CultureInfo.CurrentCulture);
            
            ;
            
            
            
            string line;
            int counter = 0;
            while((line = streamReader.ReadLine()) != null){
                row = line.Split(',');
                if (row[0] != "$Date"){
                    
                    //Console.WriteLine(row[1]);
                    spyy[counter] = Double.Parse(row[1]);
                    for(int i = 2; i < 13; i++){
                        spyx[counter,i-2] = (double)Int32.Parse(row[i]);
                    }
                    counter += 1;
                }
                

            }
            counter = 0;
            double[,] energyx = new double[19735,27];
            double[] energyy = new double[19735];
            while((line = streamReader2.ReadLine()) != null){
                row = line.Split(',');
                if (row[0] != "OUT"){
                    
                    //Console.WriteLine(row[1]);
                    energyy[counter] = (double)Int32.Parse(row[0]);
                    for(int i = 1; i < 28; i++){
                        energyx[counter,i-1] = Double.Parse(row[i]);
                    }
                    counter += 1;
                }
                

            }

            counter = 0;
            double[,] caspX = new double[45730,9];
            double[] caspY = new double[45730];
            while((line = streamReader3.ReadLine()) != null){
                row = line.Split(',');
                if (row[0] != "RMSD"){
                    
                    // /Console.WriteLine(row[0]);
                    caspY[counter] = Double.Parse(row[0]);
                    for(int i = 1; i < 10; i++){
                        caspX[counter,i-1] = Double.Parse(row[i]);
                    }
                    counter += 1;
                }
                

            }
            //e2.print2d(energyx);


            //record User(string FirstName, String LastName, string Occupation);

            //Console.ReadLine();
            //e2.print2d(spyx);
            //Console.ReadLine();
            
            Context context = Context.Create(builder => builder.AllAccelerators());
            Device dev = context.GetPreferredDevice(preferCPU: false);
            Accelerator accelerate = dev.CreateAccelerator(context);
            // //learning_rate, iterations, l1_penality, l2_penality 
            ElasticNet e1 = new ElasticNet();
            ElasticNet e3 = new ElasticNet();
            
            e1.test(6,5);
           
            // aa.Dispose();

            Random q = new Random();
            /////xCreates input data
            double counter2 = 1.0;
            double[,] Xactual = new double[10000, 100];
            for (int i = 0; i < Xactual.GetLength(0); i++)
            {
                for (int j = 0; j < Xactual.GetLength(1); j++)
                {
                    if(i %3 == 0){
                        Xactual[i, j] = counter2;
                    }
                    else{
                        if (j != 0 && i !=  0){
                            Xactual[i, j] = counter2 * i/j;
                        }
                        
                        else{
                            Xactual[i, j] = 3.5 + counter2;
                        }
                    }
                    
                    counter2 += 1.0;
                }
                //Xactual[i, 1] = ((double)q.NextDouble() * 10) + 2;

            }
            // Console.WriteLine("Xactual");
            //
            //e1.writetoCSV(Xactual, "BigData2X.csv", "IN");
            //Creates output data
            double[] Yactual = new double[10000];
            // Console.WriteLine(Xactual.GetLength(0));
            // Console.WriteLine(Xactual.GetLength(1));
            // Console.ReadLine();
            for (int i = 0; i < Yactual.GetLength(0); i++)
            {
                
                Yactual[i] = e1.sumof(Xactual, i);
                //_Console.WriteLine(Yactual[i,j]);
                
                // Yactual[i, 0] = ((Xactual[i, 0]) * 1000 + 2000);
                // Yactual[i, 1] = ((Xactual[i, 0]) * 100 + 100);
                //Yactual[i, 1] = ((Xactual[i, 0]) * 1000 + 2000);
            }
            // Console.WriteLine("Yactual");
            //e1.writetoCSV(Yactual, "BigData2Y.csv", "OUT");
            // e1.print2d(spyx);
            e1.writetoCSVFullClean(Xactual, Yactual, "Small.csv");
            ElasticNet enet = new ElasticNet();
            Stopwatch stopwtch = new Stopwatch();
            stopwtch.Start();
            enet.fit(caspX, caspY, 4.5, 4.5, 1E-100, 100);
            //double[,] res2 = e2.predict(Xactual);
            stopwtch.Stop();
           


            double[] pred =enet.predict(caspX);
            
            // Console.WriteLine("Finshed Building Data");
            // Console.WriteLine("Ypred");
            //e1.print1d(pred);

            double error = 0.0;
            //double averr = 0.0;

            for(int i = 0; i< pred.GetLength(0); i++){
               error += Math.Abs(caspY[i] - pred[i]);
            }
            Console.WriteLine();
            Console.WriteLine("average error");
            Console.WriteLine(error/pred.GetLength(0));
            Console.WriteLine();

            Console.WriteLine("TIME:");
            //e1.print1d(Yactual);
            Console.WriteLine(stopwtch.Elapsed);

            Console.WriteLine("W");
            
            e1.print1d(enet.W);
            Console.WriteLine(enet.B);


            // for(int i = 0; i < Yactual.GetLength(0); i ++){
            //     Console.Write("Actual: ");
            //     Console.Write(Yactual[i]);
            //     Console.Write(" | Pred: ");
            //     Console.Write(pred[i]);
            // }


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
            //Console.WriteLine("With accelerator");
            //Console.WriteLine(stopwatch.Elapsed);



            // Stopwatch stopwatch2 = new Stopwatch();


            // stopwatch2.Start();
            Stopwatch stopw2 = new Stopwatch();
            stopw2.Start();
            //e2.fitFULLGPU(Xactual, Yactual);
            stopw2.Stop();



            // Stopwatch stopwN = new Stopwatch();
            // stopwN.Start();
            // e3.fitNOGPU(Xactual, Yactual);
            // stopwN.Stop();
            // // stopwatch2.Stop();
            Console.WriteLine("Without accelerator");
            //

            Stopwatch stopw3 = new Stopwatch();
            stopw3.Start();
            //double[,] res2 = e2.predict(Xactual);
            stopw3.Stop();



            // Stopwatch stopwN1 = new Stopwatch();
            // stopwN1.Start();
            // double[,] resN = e3.predictNOGPU(Xactual);
            // stopwN1.Stop();

            Console.WriteLine("RES");
            // e2.print2d(res);
            Console.WriteLine("RES2");
            // e2.print2d(res2);   
            Console.WriteLine("RESN");
            // e2.print2d(resN);
            Console.WriteLine("Actual");
            // e2.print2d(Yactual);

            
        }


    }
}
