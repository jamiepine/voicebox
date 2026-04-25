import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { GpuAcceleration } from './GpuAcceleration';

// Mock dependencies
vi.mock('@/lib/api/client', () => ({
  apiClient: {
    getHealth: vi.fn(),
    getCudaStatus: vi.fn(),
    getRocmStatus: vi.fn(),
    downloadCudaBackend: vi.fn(),
    downloadRocmBackend: vi.fn(),
    deleteCudaBackend: vi.fn(),
    deleteRocmBackend: vi.fn(),
  },
}));

vi.mock('@/lib/hooks/useServer', () => ({
  useServerHealth: vi.fn(),
}));

vi.mock('@/platform/PlatformContext', () => ({
  usePlatform: vi.fn(),
}));

vi.mock('@/stores/serverStore', () => ({
  useServerStore: vi.fn((selector) => selector({ serverUrl: 'http://localhost:8000' })),
}));

import { apiClient } from '@/lib/api/client';
import { useServerHealth } from '@/lib/hooks/useServer';
import { usePlatform } from '@/platform/PlatformContext';

const mockedApiClient = vi.mocked(apiClient);
const mockedUseServerHealth = vi.mocked(useServerHealth);
const mockedUsePlatform = vi.mocked(usePlatform);

describe('GpuAcceleration', () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });

    // Reset all mocks
    vi.clearAllMocks();

    // Default platform mock (Tauri app)
    mockedUsePlatform.mockReturnValue({
      metadata: { isTauri: true },
      lifecycle: {
        restartServer: vi.fn().mockResolvedValue(undefined),
        setBackendOverride: vi.fn().mockResolvedValue(undefined),
      },
    } as any);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  function renderComponent() {
    return render(
      <QueryClientProvider client={queryClient}>
        <GpuAcceleration />
      </QueryClientProvider>,
    );
  }

  it('renders CPU status when no GPU is available', async () => {
    mockedUseServerHealth.mockReturnValue({
      data: {
        status: 'healthy',
        gpu_available: false,
        backend_variant: 'cpu',
      },
      isLoading: false,
    } as any);

    mockedApiClient.getCudaStatus.mockResolvedValue({
      available: false,
      active: false,
      downloading: false,
    });

    mockedApiClient.getRocmStatus.mockResolvedValue({
      available: false,
      active: false,
      downloading: false,
    });

    renderComponent();

    await waitFor(() => {
      expect(screen.getByText('CPU')).toBeInTheDocument();
    });
  });

  it('shows "Download AMD ROCm Backend" button when running CPU on AMD hardware', async () => {
    mockedUseServerHealth.mockReturnValue({
      data: {
        status: 'healthy',
        gpu_available: false,
        backend_variant: 'cpu',
      },
      isLoading: false,
    } as any);

    mockedApiClient.getCudaStatus.mockResolvedValue({
      available: false,
      active: false,
      downloading: false,
    });

    mockedApiClient.getRocmStatus.mockResolvedValue({
      available: false,
      active: false,
      downloading: false,
    });

    renderComponent();

    await waitFor(() => {
      expect(screen.getByText('Download AMD ROCm Backend')).toBeInTheDocument();
    });
  });

  it('shows ROCm download progress via SSE events', async () => {
    mockedUseServerHealth.mockReturnValue({
      data: {
        status: 'healthy',
        gpu_available: false,
        backend_variant: 'cpu',
      },
      isLoading: false,
    } as any);

    mockedApiClient.getCudaStatus.mockResolvedValue({
      available: false,
      active: false,
      downloading: false,
    });

    mockedApiClient.getRocmStatus.mockResolvedValue({
      available: false,
      active: false,
      downloading: true,
      download_progress: {
        model_name: 'rocm-backend',
        current: 0,
        total: 1000,
        progress: 0,
        filename: 'Downloading ROCm libraries...',
        status: 'downloading',
        timestamp: new Date().toISOString(),
      },
    });

    // Mock EventSource — use vi.stubGlobal so vi.restoreAllMocks() in afterEach
    // tears it down automatically and doesn't bleed into other tests.
    const mockEventSource = {
      onmessage: null as ((event: MessageEvent) => void) | null,
      onerror: null as (() => void) | null,
      close: vi.fn(),
    };

    vi.stubGlobal('EventSource', vi.fn(() => mockEventSource));

    renderComponent();

    await waitFor(() => {
      expect(screen.getByText('Downloading ROCm libraries...')).toBeInTheDocument();
    });

    // Simulate SSE progress update
    if (mockEventSource.onmessage) {
      mockEventSource.onmessage(
        new MessageEvent('message', {
          data: JSON.stringify({
            model_name: 'rocm-backend',
            current: 500,
            total: 1000,
            progress: 50,
            filename: 'Downloading ROCm libraries...',
            status: 'downloading',
            timestamp: new Date().toISOString(),
          }),
        }),
      );
    }

    await waitFor(() => {
      expect(screen.getByText('50.0%')).toBeInTheDocument();
    });

    // Simulate completion
    if (mockEventSource.onmessage) {
      mockEventSource.onmessage(
        new MessageEvent('message', {
          data: JSON.stringify({
            model_name: 'rocm-backend',
            current: 1000,
            total: 1000,
            progress: 100,
            filename: 'Extracting ROCm libraries...',
            status: 'complete',
            timestamp: new Date().toISOString(),
          }),
        }),
      );
    }
  });

  it('shows "Switch to CPU Backend" when running ROCm', async () => {
    mockedUseServerHealth.mockReturnValue({
      data: {
        status: 'healthy',
        gpu_available: true,
        gpu_type: 'ROCm (AMD Radeon RX 7900 XTX)',
        backend_variant: 'rocm',
        vram_used_mb: 2048,
      },
      isLoading: false,
    } as any);

    renderComponent();

    await waitFor(() => {
      expect(screen.getByText('AMD Radeon RX 7900 XTX')).toBeInTheDocument();
      expect(screen.getByText('Switch to CPU Backend')).toBeInTheDocument();
    });
  });

  it('shows "Switch to ROCm Backend" when ROCm is downloaded but not active', async () => {
    mockedUseServerHealth.mockReturnValue({
      data: {
        status: 'healthy',
        gpu_available: false,
        backend_variant: 'cpu',
      },
      isLoading: false,
    } as any);

    mockedApiClient.getCudaStatus.mockResolvedValue({
      available: false,
      active: false,
      downloading: false,
    });

    mockedApiClient.getRocmStatus.mockResolvedValue({
      available: true,
      active: false,
      downloading: false,
    });

    renderComponent();

    await waitFor(() => {
      expect(screen.getByText('Switch to ROCm Backend')).toBeInTheDocument();
    });
  });

  it('calls downloadRocmBackend when AMD download button is clicked', async () => {
    mockedUseServerHealth.mockReturnValue({
      data: {
        status: 'healthy',
        gpu_available: false,
        backend_variant: 'cpu',
      },
      isLoading: false,
    } as any);

    mockedApiClient.getCudaStatus.mockResolvedValue({
      available: false,
      active: false,
      downloading: false,
    });

    mockedApiClient.getRocmStatus.mockResolvedValue({
      available: false,
      active: false,
      downloading: false,
    });

    mockedApiClient.downloadRocmBackend.mockResolvedValue({
      message: 'ROCm backend download started',
      progress_key: 'rocm-backend',
    });

    renderComponent();

    const downloadButton = await screen.findByText('Download AMD ROCm Backend');
    fireEvent.click(downloadButton);

    await waitFor(() => {
      expect(mockedApiClient.downloadRocmBackend).toHaveBeenCalledTimes(1);
    });
  });

  it('calls setBackendOverride("cpu") when switching from ROCm to CPU', async () => {
    const setBackendOverrideMock = vi.fn().mockResolvedValue(undefined);
    mockedUsePlatform.mockReturnValue({
      metadata: { isTauri: true },
      lifecycle: {
        restartServer: vi.fn().mockResolvedValue(undefined),
        setBackendOverride: setBackendOverrideMock,
      },
    } as any);

    mockedUseServerHealth.mockReturnValue({
      data: {
        status: 'healthy',
        gpu_available: true,
        gpu_type: 'ROCm (AMD Radeon RX 7900 XTX)',
        backend_variant: 'rocm',
        vram_used_mb: 2048,
      },
      isLoading: false,
    } as any);

    renderComponent();

    const switchButton = await screen.findByText('Switch to CPU Backend');
    fireEvent.click(switchButton);

    await waitFor(() => {
      expect(setBackendOverrideMock).toHaveBeenCalledWith('cpu');
    });
  });
});
