using OpenCL.Net;
using OpenCvSharp;
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Test
{
    public class Sample
    {
        //gpu
        // GPU 메모리 할당
        [DllImport("CudaRuntime.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void create(int size);

        // CPU -> GPU 
        [DllImport("CudaRuntime.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void copyToDevice(int[] a, int[] b);

        // GPU 알고리즘 수행
        [DllImport("CudaRuntime.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void addArrays();

        // GPU -> CPU 
        [DllImport("CudaRuntime.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void copyToHost(int[] c);

        // GPU 메모리 해제 
        [DllImport("CudaRuntime.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void release();

        // CPU
        public static void addArraysParallel(int[] a, int[] b, int[] c, int size)
        {
            Parallel.For(0, size, i =>
            {
                c[i] = a[i] + b[i];
            });
        }

        public static void addArraysFor(int[] a, int[] b, int[] c, int size)
        {
            for (int i = 0; i < a.Length; i++)
            {
                c[i] = a[i] + b[i];
            }
        }

        // CV
        public static Mat _a;
        public static Mat _b;
        public static Mat _c;
        public static void addArraysOpenCV(int[] a, int[] b, int[] c, int size)
        {
            _a = new Mat(size, 1, MatType.CV_32SC1, a);
            _b = new Mat(size, 1, MatType.CV_32SC1, b);
            _c = new Mat(size, 1, MatType.CV_32SC1);
            Cv2.Add(_a, _b, _c);
            _c.GetArray(out int[] result);
            Array.Copy(result, c, size);
        }

        // OpenCL 관련 필드
        public static Platform[] platforms;
        public static Device[] devices;
        public static Context context;
        public static CommandQueue queue;
        public static Kernel kernel;
        public static Program program;
        public static IMem<int> aBuffer, bBuffer, cBuffer;
        public static void SetupOpenCL(int size)
        {
            OpenCL.Net.ErrorCode error;
            platforms = Cl.GetPlatformIDs(out error);
            devices = Cl.GetDeviceIDs(platforms[0], DeviceType.Gpu, out error);
            context = Cl.CreateContext(null, 1, devices, null, IntPtr.Zero, out error);
            queue = Cl.CreateCommandQueue(context, devices[0], CommandQueueProperties.None, out error);
            
            string kernelSource = @"
                __kernel void addArrays(__global const int* a, __global const int* b, __global int* c) {
                    int index = get_global_id(0);
                    c[index] = a[index] + b[index];
                }";

            program = Cl.CreateProgramWithSource(context, 1, new[] { kernelSource }, null, out error);
            Cl.BuildProgram(program, 1, devices, string.Empty, null, IntPtr.Zero);
            
            kernel = Cl.CreateKernel(program, "addArrays", out error);
            
            aBuffer = Cl.CreateBuffer<int>(context, MemFlags.ReadOnly, size, out error);
            bBuffer = Cl.CreateBuffer<int>(context, MemFlags.ReadOnly, size, out error);
            cBuffer = Cl.CreateBuffer<int>(context, MemFlags.WriteOnly, size, out error);
            
        }

        public static void addArraysOpenCL(int[] a, int[] b, int[] c, int size)
        {
            OpenCL.Net.ErrorCode error;

            Cl.EnqueueWriteBuffer(queue, aBuffer, Bool.True, 0, size, a, 0, null, out _);
            Cl.EnqueueWriteBuffer(queue, bBuffer, Bool.True, 0, size, b, 0, null, out _);
            
            Cl.SetKernelArg(kernel, 0, aBuffer);
            Cl.SetKernelArg(kernel, 1, bBuffer);
            Cl.SetKernelArg(kernel, 2, cBuffer);
            
            IntPtr[] globalWorkSize = new IntPtr[] { (IntPtr)size };
            Cl.EnqueueNDRangeKernel(queue, kernel, 1, null, globalWorkSize, null, 0, null, out _);
            Cl.EnqueueReadBuffer(queue, cBuffer, Bool.True, 0, size , c, 0, null, out _);
            
            Cl.Finish(queue);
        }

        public static void ReleaseOpenCL()
        {
            Cl.ReleaseMemObject(aBuffer);
            Cl.ReleaseMemObject(bBuffer);
            Cl.ReleaseMemObject(cBuffer);
            Cl.ReleaseKernel(kernel);
            Cl.ReleaseProgram(program);
            Cl.ReleaseCommandQueue(queue);
            Cl.ReleaseContext(context);
        }

        public static void Main(string[] args)
        {
            //int _size = 3 * 1600 * 1156;
            int _size = 10000000;
            int _length = 30;
            int[] _a = new int[_size];
            int[] _b = new int[_size];
            int[] _cuda = new int[_size];
            int[] _parallel = new int[_size];
            int[] _for = new int[_size];
            int[] _cv = new int[_size];
            int[] _opencl = new int[_size];

            for (int i = 0; i < _size; i++)
            {
                _a[i] = i;
                _b[i] = _size - i;
            }

            Stopwatch sw = new();

            // CPU(Parallel.For)
            Console.WriteLine("CPU Parallel.For");
            for (int i = 0; i < _length; i++)
            {
                sw.Restart();
                addArraysParallel(_a, _b, _parallel, _size);
                sw.Stop();
                Console.WriteLine($"Time: {sw.ElapsedMilliseconds} ms");
            }
            Print(_parallel);

            // CPU(For)
            Console.WriteLine("CPU For");
            for (int i = 0; i < _length; i++)
            {
                sw.Restart();
                addArraysFor(_a, _b, _for, _size);
                sw.Stop();
                Console.WriteLine($"Time: {sw.ElapsedMilliseconds} ms");
            }
            Print(_for);

            // CV
            Console.WriteLine("CPU OpenCV");
            for (int i = 0; i < _length; i++)
            {
                sw.Restart();
                addArraysOpenCV(_a, _b, _cv, _size);
                sw.Stop();
                Console.WriteLine($"Time: {sw.ElapsedMilliseconds} ms");
            }
            Print(_cv);

            // GPU(CUDA) 
            Console.WriteLine("GPU");
            create(_size);
            for (int i = 0; i < _length; i++)
            {
                sw.Restart();
                copyToDevice(_a, _b);
                addArrays();
                copyToHost(_cuda);
                sw.Stop();
                Console.WriteLine($"Time: {sw.ElapsedMilliseconds} ms");
            }
            release();
            Print(_cuda);

            //OpenCL
            Console.WriteLine("OpenCL");

            SetupOpenCL(_size);
            for (int i = 0; i < _length; i++)
            {
                sw.Restart();
                addArraysOpenCL(_a, _b, _opencl, _size);
                sw.Stop();
                Console.WriteLine($"Time: {sw.ElapsedMilliseconds} ms");
            }
            ReleaseOpenCL();
            Print(_opencl);
        }

        public static void Print(int[] value)
        {
#if DEBUG
#else
            for (int i = 0; i < 10;i++)
            {
                Console.WriteLine(value[i]);
            }
#endif

        }

    }

}
