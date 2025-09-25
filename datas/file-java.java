import java.io.IOException;
import java.io.InputStream;
import java.nio.file.*;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Instant;
import java.util.Locale;

public class SampleTool {

    public static void main(String[] args) {
        System.out.println("=== SampleTool ===");
        System.out.println("Time: " + Instant.now());
        System.out.println("Java: " + System.getProperty("java.version"));
        System.out.println("OS: " + System.getProperty("os.name") + " " + System.getProperty("os.version"));
        System.out.println("User: " + System.getProperty("user.name"));
        System.out.println();

        if (args.length == 0) {
            System.out.println("Usage: java SampleTool <path-to-file>");
            System.out.println("No file provided. Showing a quick demo:");
            System.out.println("  -> Hello from SampleTool! ✨");
            return;
        }

        Path file = Paths.get(args[0]);
        if (!Files.exists(file)) {
            System.err.println("File not found: " + file.toAbsolutePath());
            System.exit(1);
        }

        try {
            String hash = sha256Hex(file);
            long size = Files.size(file);
            long lines = safeCountLines(file);

            System.out.println("Analyzed file: " + file.toAbsolutePath());
            System.out.println(" - Size:  " + size + " bytes");
            System.out.println(" - Lines: " + lines);
            System.out.println(" - SHA-256: " + hash);
        } catch (IOException e) {
            System.err.println("I/O error: " + e.getMessage());
            System.exit(2);
        } catch (NoSuchAlgorithmException e) {
            System.err.println("Hash algorithm not available: " + e.getMessage());
            System.exit(3);
        }
    }

    private static String sha256Hex(Path file) throws IOException, NoSuchAlgorithmException {
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        try (InputStream in = Files.newInputStream(file);
             DigestInputStream dis = new DigestInputStream(in, md)) {

            byte[] buffer = new byte[8192];
            while (dis.read(buffer) != -1) {
                // digest is updated by DigestInputStream
            }
        }
        return toHex(md.digest());
    }

    private static long safeCountLines(Path file) {
        try {
            // Works fine for text-ish files; for binary files line count is not meaningful but harmless
            try (var lines = Files.lines(file)) {
                return lines.count();
            }
        } catch (IOException e) {
            return -1; // indicate unknown
        }
    }

    private static String toHex(byte[] bytes) {
        StringBuilder sb = new StringBuilder(bytes.length * 2);
        for (byte b : bytes) sb.append(String.format(Locale.ROOT, "%02x", b));
        return sb.toString();
    }
}
