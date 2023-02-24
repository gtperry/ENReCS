
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
            
            //double lambda2 = 0.5f;
            
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
                double, double, int>(
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
                double>(DualityGapKernal);

            var PobjKern = this.accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                double>(PobjKernal);

            var DobjKern = this.accelerate.LoadAutoGroupedStreamKernel<Index1D,
                ArrayView1D<double, Stride1D.Dense>,
                ArrayView1D<double, Stride1D.Dense>,
                double>(DobjKernal);

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
            
            
            using (this.ColMeansBuffer)
            using (this.XBuffer)
            {
                
                //Calculate the mean of each column of X
                columnMeansKern(this.ColMeansBuffer.Extent.ToIntIndex(), this.XBuffer.View, this.ColMeansBuffer.View, X.GetLength(0));
                //Calculate the std dev of each column of X
                columnSTDevKern(this.ColSTDBuffer.Extent.ToIntIndex(), this.XBuffer.View, this.ColSTDBuffer.View, this.ColMeansBuffer.View, X.GetLength(0));
                
                //Fill x2
                x2fillkern(new Index1D(X.GetLength(1)), this.XBuffer.View, this.X2Buffer.View, this.ColMeansBuffer.View, this.ColSTDBuffer.View, c, padding, X.GetLength(0));
                
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
                    DualityGapKern(this.NuBuffer.Extent.ToIntIndex(), this.NuBuffer.View, this.MaxValBuffer.View, lambda);

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
            //Have all threads catch up to each other
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
        
        
        static void SMSumOfKernal(
            ArrayView<double> dataView,          
            ArrayView<double> outputView,
            int gridsize)        
        {
            ///<summary>Calculates the sum of an ArrayView</summary>
            ///<param name="aView">The ArrayView</param>
            ///<param name="outputView">The sum</param>
            ///<param name="gridsize">The number of elements in each grid</param>
            
            int globalIndex = Grid.LinearIndex;
            
            int localindex = Group.LinearIndex;

            // 'Allocate' a single shared memory variable of type int (= 4 bytes)
            ref double sharedVariable = ref ILGPU.SharedMemory.Allocate<double>();

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
            ArrayView2D<double, Stride2D.DenseX> dataView,          
            ArrayView<double> outputView,
            int gridsize)        
        {
            ///<summary>Calculates the sum of the log(-x) of all elements an ArrayView</summary>
            ///<param name="aView">The ArrayView</param>
            ///<param name="outputView">The sum</param>
            ///<param name="gridsize">The number of elements in each grid</param>

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
            ///<summary>Used to comp if the newphi has reached a stopping point, set output to 1 if it has</summary>
            ///<param name="phi">Phi</param>
            ///<param name="newphi">New Phi</param>
            ///<param name="gradxDxu">GDX</param>
            ///<param name="Output">The output</param>
            ///<param name="alpha">constant</param>
            ///<param name="s">s value</param>
            if(newphi[index] - phi[index] <= alpha * s * gradxDxu[index]){
                output[index] = 1;

            }
            else{
                output[index] = 0;
            }
           

        }
        static void WFinaleKernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> wView,
            ArrayView1D<double, Stride1D.Dense> scale,
            double c
            ){
            ///<summary>De-Normalizes W after all computations</summary>
            ///<param name="wView">W Values</param>
            ///<param name="scale">Std devs of the columns of the data</param>
            ///<param name="c">variable based on lambda2</param>
            wView[index] = (c * wView[index]) / scale[index];

        }
        static void SubYFromZKernal(
            Index1D index,
            ArrayView1D<double, Stride1D.Dense> zView,
            ArrayView1D<double, Stride1D.Dense> YView)
        {
            ///<summary>Subtracts the elements of Yview from the corresponding elements in Zview</summary>
            ///<param name="zView">zView</param>
            ///<param name="yView">yVuew</param>
            zView[index] = zView[index] - YView[index];
        }
        
        static void CalcphiKernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> phi,
            ArrayView1D<double, Stride1D.Dense> zdot,
            ArrayView1D<double, Stride1D.Dense> usum,
            ArrayView1D<double, Stride1D.Dense> fsumlgneg,
            double t, double lambda)
        {
            ///<summary>Calculates the phi value</summary>
            ///<param name="phi">phi</param>
            ///<param name="zdot">zdot</param>
            ///<param name="usum">usum</param>
            ///<param name="fsumlgneg">fsumlgneg</param>
            ///<param name="t">t</param>
            ///<param name="lambda">lambda</param>
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
            ///<summary>Sets the new values of the w,u, and f buffers. Also switches fmax to positive if there is a positive value in newf</summary>
            ///<param name="wView">old w buffer</param>
            ///<param name="uView">old u buffer</param>
            ///<param name="dxView">dx buffer, used for calculations</param>
            ///<param name="duView">du buffer, used for calculations</param>
            ///<param name="newwView">the new w buffer</param>
            ///<param name="newuView">the new u buffer</param>
            ///<param name="newfView">the new f buffer</param>
            ///<param name="fmax">fmax</param>
            ///<param name="s">s</param>
            
            newwView[index] = wView[index] + s * dxView[index];
            newuView[index] = uView[index] + s * duView[index];
            newfView[new Index2D(0, index.X)] = newwView[index] - newuView[index];
            newfView[new Index2D(1, index.X)] = (-1.0 * newwView[index]) - newuView[index];
            if(newfView[new Index2D(0, index.X)] > 0 || newfView[new Index2D(1, index.X)] > 0){
                fmax[0] = 1.0;
            }

        }

        
        static void splitDXUKernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> dxView,
            ArrayView1D<double, Stride1D.Dense> duView,
            ArrayView1D<double, Stride1D.Dense> dxuView,
            int p)
        {
            ///<summary>Fills DX and DU with the front and back halves of dxu respectively</summary>
            ///<param name="dxView">dx</param>
            ///<param name="duView">du</param>
            ///<param name="dxuView">dxu</param>
            ///<param name="p">length of dx</param>
            dxView[index] = dxuView[index];
            duView[index] = dxuView[new Index1D(index.X + p)];

        }
        static void calcErrorKernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> rnorm,
            ArrayView1D<double, Stride1D.Dense> bnorm,
            ArrayView1D<double, Stride1D.Dense> error
            )
        {
            ///<summary>Calculates the error from rnorm and bnorm</summary>
            ///<param name="rnorm">rnorm</param>
            ///<param name="bnorm">bnorm</param>
            ///<param name="error">error</param>
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
            ///<summary>Adjusts the rView, rrView, and xView</summary>
            ///<param name="rView">rView</param>
            ///<param name="rrView">rrView</param>
            ///<param name="zView">zView</param>
            ///<param name="zzView">zzView</param>
            ///<param name="bknumView">bknumView</param>
            ///<param name="akdenView">akdenView</param>
            ///<param name="xView">xView</param>
            ///<param name="pView">pView</param>
            
            rView[index] -= (bknumView[new Index1D(0)] / akdenView[new Index1D(0)]) * zView[index];
            rrView[index] -= (bknumView[new Index1D(0)] / akdenView[new Index1D(0)]) * zzView[index];
            xView[index] += (bknumView[new Index1D(0)] / akdenView[new Index1D(0)]) * pView[index];

        }

        static void CopyBufferKernal1D(Index1D index,
            ArrayView1D<double, Stride1D.Dense> aView,
            ArrayView1D<double, Stride1D.Dense> bView)

        {
            ///<summary>Copys aView into Bview</summary>
            ///<param name="aView">aView</param>
            ///<param name="bView">bView</param>
            bView[index] = aView[index];

        }
        static void CopyBufferKernal2D(Index2D index,
            ArrayView2D<double, Stride2D.DenseX> aView,
            ArrayView2D<double, Stride2D.DenseX> bView)
        {
            ///<summary>Copys aView into Bview</summary>
            ///<param name="aView">aView</param>
            ///<param name="bView">bView</param>
            bView[index] = aView[index];

        }

        static void FirstIterationFillPsKernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> pView,
            ArrayView1D<double, Stride1D.Dense> ppView,
            ArrayView1D<double, Stride1D.Dense> zView,
            ArrayView1D<double, Stride1D.Dense> zzView)
        {
            ///<summary>Fills p with z and pp with zz</summary>
            ///<param name="pView">pView</param>
            ///<param name="ppView">ppView</param>
            ///<param name="zView">zView</param>
            ///<param name="zzView">zzView</param>
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
            ///<summary>Adjusts the pView and ppView</summary>
            ///<param name="pView">pView</param>
            ///<param name="ppView">ppView</param>
            ///<param name="zView">zView</param>
            ///<param name="zzView">zzView</param>
            ///<param name="bknumView">bknumView</param>
            ///<param name="bkdenView">bkdenView</param>

            pView[index] = ((bknumView[new Index1D(0)] / bkdenView[new Index1D(0)]) * pView[index]) + zView[index];
            ppView[index] = ((bknumView[new Index1D(0)] / bkdenView[new Index1D(0)]) * ppView[index]) + zzView[index];

        }


        static void fillInverseMatrixKernal(Index2D index,
            ArrayView2D<double, Stride2D.DenseX> aView,
            ArrayView2D<double, Stride2D.DenseX> inverseView)
        {
            ///<summary>Fills p with z and pp with zz</summary>
            ///<param name="pView">pView</param>
            ///<param name="ppView">ppView</param>

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
            ///<summary>Asolve algorithm, adjusts xView</summary>
            ///<param name="bView">bView</param>
            ///<param name="xView">xView</param>
            ///<param name="d1View">d1View</param>
            ///<param name="d2View">d2View</param>
            ///<param name="prsView">prsView</param>
            ///<param name="prbView">prbView</param>
            ///<param name="p"> half of length of xView</param>
            
            xView[index] = ((d1View[index] * bView[index]) - (d2View[index] * bView[new Index1D(index.X + p)])) / prsView[index];
            xView[new Index1D(index.X + p)] = ((-d2View[index] * bView[index]) + (prbView[index] * bView[new Index1D(index.X + p)])) / prsView[index];

        }
        static void RandRRFillKernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> bView,
            ArrayView1D<double, Stride1D.Dense> rView,
            ArrayView1D<double, Stride1D.Dense> rrView
            )
        {
            ///<summary>Fills r and rr using bView</summary>
            ///<param name="bView">bView</param>
            ///<param name="rView">rView</param>
            ///<param name="rrView">rrView</param>

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
            ///<summary>Mv algo that replaces matrix multiplication in PCG algo</summary>
            ///<param name="ataxView">ataxView</param>
            ///<param name="xView">xView</param>
            ///<param name="yView">yView</param>
            ///<param name="d1View">d1View</param>
            ///<param name="d2View">d2View</param>
            ///<param name="p">half of length of yView</param>

            int z = index.X;
           
            yView[index] = (2.0 * ataxView[index]) + (d1View[index] * xView[index]) + (d2View[index] * xView[new Index1D(index.X + p)]);
            yView[new Index1D(index.X + p)] = (d2View[index] * xView[index]) + (d1View[index] * xView[new Index1D(index.X + p)]);



        }
        
        static void PreconditionerVectorKernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> prbView,
            ArrayView1D<double, Stride1D.Dense> prsView,
            ArrayView1D<double, Stride1D.Dense> d1View,
            ArrayView1D<double, Stride1D.Dense> d2View,
            ArrayView1D<double, Stride1D.Dense> diagxtxView)
        {
            ///<summary>Calculates preconditioner vector variables</summary>
            ///<param name="prbView">prbView</param>
            ///<param name="prsView">prsView</param>
            ///<param name="d1View">d1View</param>
            ///<param name="d2View">d2View</param>
            ///<param name="diagxtxView">diagxtxView</param>
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
            ///<summary>Calculates gradient</summary>
            ///<param name="GradPhiView">GradPhiView</param>
            ///<param name="q1View">q1View</param>
            ///<param name="q2View">q2View</param>
            ///<param name="gradView">gradView</param>
            ///<param name="t">t variable</param>
            ///<param name="lambda">lambda variable</param>
            ///<param name="p">length of half of gradview</param>
            GradPhiView[new Index2D(0, index.X)] = (2.0 * GradPhiView[new Index2D(0, index.X)]) - ((q1View[index] - q2View[index]) / t);
            GradPhiView[new Index2D(1, index.X)] = lambda - (q1View[index] + q2View[index]) / t;//
            gradView[index] = -1.0 * GradPhiView[new Index2D(0, index.X)];
            gradView[new Index1D(index.X + p)] = -1.0 * GradPhiView[new Index2D(1, index.X)];

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
            ///<summary>Calculates newton step</summary>
            ///<param name="uView">uView</param>
            ///<param name="wView">wView</param>
            ///<param name="q1View">q1View</param>
            ///<param name="q2View">q2View</param>
            ///<param name="d1View">d1View</param>
            ///<param name="d2View">d2View</param>
            ///<param name="t">t variable</param>

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
            double lambda)
        {
            ///<summary>Calculates duality gap</summary>
            ///<param name="NuView">NuView</param>
            ///<param name="MaxXnu">MaxXnu</param>
            ///<param name="lambda">lambda</param>
            
            double mxxnu = MaxXnu[new Index1D(0)];

            if (mxxnu > lambda)
            {
                double lnu = lambda / mxxnu;
                
                NuView[index] = NuView[index] * lnu;
            }

        }
        
        
        static void sqrtkernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> aView)
        {
            ///<summary>Calculates the sqrt of all values in a buffer</summary>
            ///<param name="aView">aView</param>

            aView[index] = Math.Sqrt(aView[index]);
        }
        
    
        static void PobjKernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> dot,
            ArrayView1D<double, Stride1D.Dense> norm1,
            double l)
        {
            ///<summary>Calculates primal objective function value</summary>
            ///<param name="dot">dot</param>
            ///<param name="norm1">norm1 input, and used as output to save space</param>
            ///<param name="l">lamda</param>

            norm1[index] = dot[index] + (l * norm1[index]);

        }
        static void DobjKernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> dot1,
            ArrayView1D<double, Stride1D.Dense> dot2,
            double dobj)
        {
            ///<summary>Calculates dual objective function value</summary>
            ///<param name="dot1">dot1 value and output to save space</param>
            ///<param name="dot2">dot2</param>
            ///<param name="dobj">previous dobj value</param>
            
            dot1[index] = Math.Max((-0.25 * dot1[index]) - dot2[index], dobj);

        }
        static void InitializeNuKernal(
            Index1D index,
            ArrayView1D<double, Stride1D.Dense> YView,
            ArrayView1D<double, Stride1D.Dense> zView,
            ArrayView1D<double, Stride1D.Dense> NuView)
        {
            ///<summary>Initializes Nu Buffer, and adjusts z based on Y</summary>
            ///<param name="YView">YView</param>
            ///<param name="zView">zView</param>
            ///<param name="NuView">NuView</param>
            zView[index] = zView[index] - YView[index];
            NuView[index] = 2.0 * zView[index];

        }

        static void GetMaxValKernal(Index1D index,
            ArrayView1D<double, Stride1D.Dense> aView,
            ArrayView1D<double, Stride1D.Dense> MaxVal,
            int aViewLength)
        {
            ///<summary>Gets the maximum value of aView</summary>
            ///<param name="aView">aView</param>
            ///<param name="MaxVal">MaxVal</param>
            ///<param name="aViewLength">length of aView</param>

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
            ///<summary>Fills the f buffer</summary>
            ///<param name="uView">uView</param>
            ///<param name="wView">wView</param>
            ///<param name="fView">fView</param>
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
            ///<summary>Gets the means of all the columns in aView</summary>
            ///<param name="aView">aView</param>
            ///<param name="bView">bView</param>
            ///<param name="y">the length of the columns</param>
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
            ///<summary>Gets the means of all the columns in aView</summary>
            ///<param name="aView">aView</param>
            ///<param name="bView">bView</param>
            ///<param name="y">the length of the columns</param>

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
            ///<summary>Subtracts every element in aView by the mean of aView</summary>
            ///<param name="aView">aView</param>
            ///<param name="bView">bView</param>
            ///<param name="meanView">the mean of aView</param>
            bView[index] = aView[index] - meanView[new Index1D(0)];
           
        }
        static void columnSTDevKernal(
            Index1D index,
            ArrayView2D<double, Stride2D.DenseX> aView,
            ArrayView1D<double, Stride1D.Dense> bView,
            ArrayView1D<double, Stride1D.Dense> meanView,
            int y
            )
        {
            ///<summary>Gets the std deviations of all the columns in aView</summary>
            ///<param name="aView">aView</param>
            ///<param name="bView">bView</param>
            ///<param name="meanView">meanView</param>
            ///<param name="y">the length of the columns</param>
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
            int m
            )
        {
            ///<summary>Fills X2 Buffer</summary>
            ///<param name="aView">aView</param>
            ///<param name="bView">bView</param>
            ///<param name="colMeans">colMeans</param>
            ///<param name="colSTDs">colSTDs</param>
            ///<param name="c">C</param>
            ///<param name="padding">padding</param>
            ///<param name="m">length of aView</param>
            
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
            ///<summary>Subtracts subvalue from every element of aView and puts result into BView</summary>
            ///<param name="aView">aView</param>
            ///<param name="bView">bView</param>
            ///<param name="subvalue">subvalue</param>
            bView[index] = aView[index] - subvalue;
        }
        

        static void SMMatMul2DKernal(Index3D index,
            ArrayView2D<double, Stride2D.DenseX> aView,
            ArrayView2D<double, Stride2D.DenseX> bView,
            ArrayView2D<double, Stride2D.DenseX> cView){
            ///<summary>Does matrix multiplication between two 2d arrays</summary>
            ///<param name="aView">aView</param>
            ///<param name="bView">bView</param>
            ///<param name="cView">output</param>
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
            ///<summary>Does matrix multiplication between a 2d and a 1d array</summary>
            ///<param name="aView">aView</param>
            ///<param name="bView">bView</param>
            ///<param name="cView">output</param>
            var x = index.X;
            var y = index.Y;
            
            
            double val = aView[new Index2D(x,y)] * bView[new Index1D(y)];
            Atomic.Add(ref cView[new Index1D(x)], val);


        }
        
        static void SMMatMul1DKernalTooLarge(Index2D index,
            ArrayView2D<double, Stride2D.DenseX> aView,
            ArrayView1D<double, Stride1D.Dense> bView,
            ArrayView1D<double, Stride1D.Dense> cView){
            ///<summary>Does matrix multiplication between a 2d and a 1d array, but adjusted for when aview dimensions are too large</summary>
            ///<param name="aView">aView</param>
            ///<param name="bView">bView</param>
            ///<param name="cView">output</param>

            var x = index.X;
            var y = index.Y;
            
            
            double val = aView[new Index2D(y,x)] * bView[new Index1D(x)];
            Atomic.Add(ref cView[new Index1D(y)], val);


        }
        static void MatrixMultiplyGradphiKernel(
            Index1D index,
            ArrayView2D<double, Stride2D.DenseX> aView,
            ArrayView1D<double, Stride1D.Dense> bView,
            ArrayView2D<double, Stride2D.DenseX> cView)
        {
            ///<summary> Does Matrix Multiplication on two arrayviews, and then stores in gradphi </summary>
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

        static void setBuffToValueKernal(Index1D index, 
            ArrayView1D<double, Stride1D.Dense> buff, 
            double setvalue)
        {
            ///<summary>Sets every element in buff to setvalue</summary>
            ///<param name="buff">buff</param>
            ///<param name="setvalue">setvalue</param>
            buff[index] = setvalue;
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
        //TEST FUNCTIONS BELOW
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
        


        static void Main(string[] args)
        {

        }


    }
}
