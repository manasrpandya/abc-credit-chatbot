const isDev = process.env.NODE_ENV === "development";

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",
  trailingSlash: true,
  images: { unoptimized: true },
  // during next dev the python api runs on its own port
  ...(isDev
    ? {
        output: undefined,
        rewrites: async () => [{ source: "/api/:path*", destination: "http://127.0.0.1:8000/api/:path*" }],
      }
    : {}),
};

export default nextConfig;
