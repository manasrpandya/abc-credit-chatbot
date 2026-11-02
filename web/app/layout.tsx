import type { Metadata, Viewport } from "next";
import { Bricolage_Grotesque, Instrument_Sans } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const display = Bricolage_Grotesque({ subsets: ["latin", "latin-ext"], variable: "--font-display" });
const body = Instrument_Sans({ subsets: ["latin", "latin-ext"], variable: "--font-body" });

export const metadata: Metadata = {
  title: "ABC Credit two-wheeler loans",
  description: "Apply for a two-wheeler loan in a short chat and get an answer straight away.",
};

export const viewport: Viewport = { themeColor: "#100d0b" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable}`}>
      <body>
        <header className="top">
          <Link href="/" className="brand">ABC Credit</Link>
          <nav>
            <Link href="/how-it-decides/">How it decides</Link>
            <Link href="/audit/">Audit log</Link>
          </nav>
        </header>
        {children}
      </body>
    </html>
  );
}
