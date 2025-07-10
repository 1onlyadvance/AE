// PROPRIETARY AND CONFIDENTIAL
// Copyright (c) 2024 DELFICTUS I/O LLC
// Patent Pending - Application #63/826,067

using UnrealBuildTool;
using System.IO;

public class ARESEdgePlugin : ModuleRules
{
    public ARESEdgePlugin(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
        
        // Enable C++20 and exceptions
        CppStandard = CppStandardVersion.Cpp20;
        bEnableExceptions = true;
        bUseRTTI = true;
        
        // Public dependencies
        PublicDependencyModuleNames.AddRange(new string[] {
            "Core",
            "CoreUObject",
            "Engine",
            "RenderCore",
            "RHI",
            "RHICore",
            "Projects",
            "HeadMountedDisplay",
            "OpenXRHMD",
            "OpenXR",
            "AugmentedReality"
        });
        
        // Private dependencies
        PrivateDependencyModuleNames.AddRange(new string[] {
            "Slate",
            "SlateCore",
            "HTTP",
            "Json",
            "JsonUtilities",
            "WebSockets"
        });
        
        // CUDA configuration
        string CudaPath = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7";
        
        PublicIncludePaths.Add(Path.Combine(CudaPath, "include"));
        PublicAdditionalLibraries.Add(Path.Combine(CudaPath, "lib/x64/cudart_static.lib"));
        PublicAdditionalLibraries.Add(Path.Combine(CudaPath, "lib/x64/cublas.lib"));
        PublicAdditionalLibraries.Add(Path.Combine(CudaPath, "lib/x64/cudnn.lib"));
        PublicAdditionalLibraries.Add(Path.Combine(CudaPath, "lib/x64/cufft.lib"));
        PublicAdditionalLibraries.Add(Path.Combine(CudaPath, "lib/x64/cusparse.lib"));
        PublicAdditionalLibraries.Add(Path.Combine(CudaPath, "lib/x64/cusolver.lib"));
        PublicAdditionalLibraries.Add(Path.Combine(CudaPath, "lib/x64/nccl.lib"));
        
        // Cryptography
        PublicAdditionalLibraries.Add("libcrypto.lib");
        PublicAdditionalLibraries.Add("libssl.lib");
        
        // Add ARES source files
        PublicIncludePaths.Add(Path.Combine(ModuleDirectory, "../../.."));
        
        // Platform specific
        if (Target.Platform == UnrealTargetPlatform.Win64)
        {
            PublicDefinitions.Add("ARES_PLATFORM_WINDOWS=1");
            PublicAdditionalLibraries.Add("ws2_32.lib");
            PublicAdditionalLibraries.Add("winmm.lib");
        }
        
        // Optimization flags
        if (Target.Configuration == UnrealTargetConfiguration.Shipping)
        {
            PublicDefinitions.Add("ARES_SHIPPING_BUILD=1");
            OptimizeCode = CodeOptimization.InShippingBuildsOnly;
        }
    }
}