import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Data Annotation Runs",
  description: "Compare OpenAI annotation script runs and confusion matrices.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
