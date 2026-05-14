import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

/** @type {import('next').NextConfig} */
const config = {
  output: 'export',
  basePath: '/voicebox',
  images: { unoptimized: true },
  reactStrictMode: true,
  ...(!process.env.NEXT_STATIC_EXPORT && {
    async rewrites() {
      return [
        {
          source: '/docs/:path*.mdx',
          destination: '/llms.mdx/docs/:path*',
        },
      ];
    },
  }),
  webpack: (config) => {
    config.experiments = {
      ...config.experiments,
      topLevelAwait: true,
    };
    return config;
  },
};

export default withMDX(config);
