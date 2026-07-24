import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "HyperTune AI — World-Class Enterprise LLM Platform",
  description: "Production-grade platform for LLM fine-tuning, evaluation, versioning, deployment, and real-time GPU telemetry.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} dark h-full antialiased`}
    >
      <body className="min-h-full bg-[#09090B] text-[#F8FAFC] flex flex-col font-sans selection:bg-[#8B5CF6]/30 selection:text-[#F8FAFC]">
        {children}
      </body>
    </html>
  );
}
