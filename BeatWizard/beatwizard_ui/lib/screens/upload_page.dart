import 'dart:typed_data';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:file_picker/file_picker.dart';
import '../services/beat_service.dart';

class UploadPage extends StatefulWidget {
  const UploadPage({super.key});

  @override
  State<UploadPage> createState() => _UploadPageState();
}

class _UploadPageState extends State<UploadPage> with TickerProviderStateMixin {
  final _titleController = TextEditingController();
  final _descriptionController = TextEditingController();
  final _beatService = BeatService();
  final _formKey = GlobalKey<FormState>();
  
  Uint8List? _selectedFileBytes;
  String? _selectedFileName;
  bool _isUploading = false;
  double _uploadProgress = 0.0;
  late AnimationController _pulseController;
  late AnimationController _waveController;

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      duration: const Duration(seconds: 2),
      vsync: this,
    )..repeat(reverse: true);
    
    _waveController = AnimationController(
      duration: const Duration(seconds: 3),
      vsync: this,
    )..repeat();
  }

  @override
  void dispose() {
    _titleController.dispose();
    _descriptionController.dispose();
    _pulseController.dispose();
    _waveController.dispose();
    super.dispose();
  }

  Future<void> _pickFile() async {
    try {
      FilePickerResult? result = await FilePicker.platform.pickFiles(
        type: FileType.audio,
        allowMultiple: false,
      );

      if (result != null) {
        setState(() {
          _selectedFileBytes = result.files.first.bytes;
          _selectedFileName = result.files.first.name;
        });
      }
    } catch (e) {
      _showError('Failed to pick file: $e');
    }
  }

  Future<void> _uploadBeat() async {
    if (!_formKey.currentState!.validate() || _selectedFileBytes == null) {
      _showError('Please fill all fields and select a beat file');
      return;
    }

    setState(() {
      _isUploading = true;
      _uploadProgress = 0.0;
    });

    try {
      // Simulate progress for better UX
      for (int i = 0; i <= 100; i += 10) {
        await Future.delayed(const Duration(milliseconds: 100));
        setState(() {
          _uploadProgress = i / 100.0;
        });
      }

      await _beatService.uploadBeat(
        title: _titleController.text,
        description: _descriptionController.text,
        fileBytes: _selectedFileBytes!,
        fileName: _selectedFileName!,
      );

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: const Row(
              children: [
                Icon(Icons.check_circle, color: Colors.white),
                SizedBox(width: 12),
                Text('Beat uploaded successfully! 🎉'),
              ],
            ),
            backgroundColor: Colors.green.shade600,
            behavior: SnackBarBehavior.floating,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          ),
        );

        // Reset form
        _titleController.clear();
        _descriptionController.clear();
        setState(() {
          _selectedFileBytes = null;
          _selectedFileName = null;
        });
      }
    } catch (e) {
      if (mounted) {
        _showError('Upload failed: $e');
      }
    } finally {
      if (mounted) {
        setState(() {
          _isUploading = false;
          _uploadProgress = 0.0;
        });
      }
    }
  }

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Row(
          children: [
            const Icon(Icons.error_outline, color: Colors.white),
            const SizedBox(width: 12),
            Expanded(child: Text(message)),
          ],
        ),
        backgroundColor: Colors.red.shade600,
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [
            Colors.transparent,
            Colors.black.withOpacity(0.1),
          ],
        ),
      ),
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Form(
          key: _formKey,
          child: Column(
            children: [
              // Header Section
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(32),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [
                      Colors.white.withOpacity(0.15),
                      Colors.white.withOpacity(0.05),
                    ],
                  ),
                  borderRadius: BorderRadius.circular(28),
                  border: Border.all(color: Colors.white.withOpacity(0.2)),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.1),
                      blurRadius: 20,
                      spreadRadius: 5,
                    ),
                  ],
                ),
                child: Column(
                  children: [
                    Container(
                      padding: const EdgeInsets.all(16),
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                          colors: [
                            Theme.of(context).colorScheme.primary.withOpacity(0.3),
                            Theme.of(context).colorScheme.primary.withOpacity(0.1),
                          ],
                        ),
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Icon(
                        Icons.cloud_upload_outlined,
                        size: 48,
                        color: Theme.of(context).colorScheme.primary,
                      ),
                    ),
                    const SizedBox(height: 16),
                    ShaderMask(
                      shaderCallback: (bounds) => LinearGradient(
                        colors: [
                          Colors.white,
                          Theme.of(context).colorScheme.primary,
                          Colors.white,
                        ],
                      ).createShader(bounds),
                      child: Text(
                        'Share Your Beat',
                        style: Theme.of(context).textTheme.headlineLarge?.copyWith(
                              color: Colors.white,
                              fontWeight: FontWeight.w800,
                              fontSize: 32,
                            ),
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'Upload your masterpiece to the world',
                      style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                            color: Colors.white.withOpacity(0.7),
                            fontSize: 16,
                          ),
                    ),
                  ],
                ),
              ).animate().fadeIn().slideY(begin: -0.3),

              const SizedBox(height: 32),

              // File Upload Section
              AnimatedBuilder(
                animation: _pulseController,
                builder: (context, child) {
                  return Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(32),
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        colors: [
                          Colors.white.withOpacity(0.1 + _pulseController.value * 0.05),
                          Colors.white.withOpacity(0.05 + _pulseController.value * 0.03),
                        ],
                      ),
                      borderRadius: BorderRadius.circular(24),
                      border: Border.all(
                        color: _selectedFileBytes != null 
                            ? Theme.of(context).colorScheme.primary.withOpacity(0.5)
                            : Colors.white.withOpacity(0.1 + _pulseController.value * 0.1),
                        width: 2,
                      ),
                      boxShadow: [
                        BoxShadow(
                          color: _selectedFileBytes != null
                              ? Theme.of(context).colorScheme.primary.withOpacity(0.2)
                              : Colors.black.withOpacity(0.05),
                          blurRadius: 20,
                          spreadRadius: 5,
                        ),
                      ],
                    ),
                    child: Column(
                      children: [
                        if (_selectedFileBytes == null) ...[
                          // Empty state with beautiful animation
                          GestureDetector(
                            onTap: _pickFile,
                            child: Container(
                              height: 200,
                              width: double.infinity,
                              decoration: BoxDecoration(
                                gradient: LinearGradient(
                                  colors: [
                                    Colors.white.withOpacity(0.05),
                                    Colors.white.withOpacity(0.02),
                                  ],
                                ),
                                borderRadius: BorderRadius.circular(20),
                                border: Border.all(
                                  color: Colors.white.withOpacity(0.1),
                                  style: BorderStyle.solid,
                                ),
                              ),
                              child: Stack(
                                children: [
                                  // Animated waves
                                  AnimatedBuilder(
                                    animation: _waveController,
                                    builder: (context, child) {
                                      return Positioned.fill(
                                        child: CustomPaint(
                                          painter: WavePainter(
                                            progress: _waveController.value,
                                            color: Theme.of(context).colorScheme.primary.withOpacity(0.1),
                                          ),
                                        ),
                                      );
                                    },
                                  ),
                                  
                                  Center(
                                    child: Column(
                                      mainAxisAlignment: MainAxisAlignment.center,
                                      children: [
                                        Icon(
                                          Icons.cloud_upload_outlined,
                                          size: 64,
                                          color: Colors.white.withOpacity(0.7),
                                        ),
                                        const SizedBox(height: 16),
                                        Text(
                                          'Drag & Drop Your Beat',
                                          style: Theme.of(context).textTheme.titleLarge?.copyWith(
                                                color: Colors.white,
                                                fontWeight: FontWeight.w600,
                                              ),
                                        ),
                                        const SizedBox(height: 8),
                                        Text(
                                          'or tap to browse files',
                                          style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                                                color: Colors.white.withOpacity(0.6),
                                              ),
                                        ),
                                        const SizedBox(height: 16),
                                        Container(
                                          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
                                          decoration: BoxDecoration(
                                            gradient: LinearGradient(
                                              colors: [
                                                Theme.of(context).colorScheme.primary.withOpacity(0.2),
                                                Theme.of(context).colorScheme.primary.withOpacity(0.1),
                                              ],
                                            ),
                                            borderRadius: BorderRadius.circular(25),
                                            border: Border.all(
                                              color: Theme.of(context).colorScheme.primary.withOpacity(0.3),
                                            ),
                                          ),
                                          child: Text(
                                            'MP3, WAV, M4A supported',
                                            style: Theme.of(context).textTheme.bodySmall?.copyWith(
                                                  color: Theme.of(context).colorScheme.primary,
                                                  fontWeight: FontWeight.w500,
                                                ),
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ),
                        ] else ...[
                          // File selected state
                          Container(
                            padding: const EdgeInsets.all(24),
                            decoration: BoxDecoration(
                              gradient: LinearGradient(
                                colors: [
                                  Theme.of(context).colorScheme.primary.withOpacity(0.15),
                                  Theme.of(context).colorScheme.primary.withOpacity(0.05),
                                ],
                              ),
                              borderRadius: BorderRadius.circular(16),
                              border: Border.all(
                                color: Theme.of(context).colorScheme.primary.withOpacity(0.3),
                              ),
                            ),
                            child: Row(
                              children: [
                                Container(
                                  padding: const EdgeInsets.all(12),
                                  decoration: BoxDecoration(
                                    gradient: LinearGradient(
                                      colors: [
                                        Theme.of(context).colorScheme.primary.withOpacity(0.3),
                                        Theme.of(context).colorScheme.primary.withOpacity(0.1),
                                      ],
                                    ),
                                    borderRadius: BorderRadius.circular(12),
                                  ),
                                  child: Icon(
                                    Icons.audiotrack,
                                    color: Theme.of(context).colorScheme.primary,
                                    size: 32,
                                  ),
                                ),
                                const SizedBox(width: 16),
                                Expanded(
                                  child: Column(
                                    crossAxisAlignment: CrossAxisAlignment.start,
                                    children: [
                                      Text(
                                        _selectedFileName!,
                                        style: Theme.of(context).textTheme.titleMedium?.copyWith(
                                              color: Colors.white,
                                              fontWeight: FontWeight.w600,
                                            ),
                                        overflow: TextOverflow.ellipsis,
                                      ),
                                      const SizedBox(height: 4),
                                      Text(
                                        'Ready to upload',
                                        style: Theme.of(context).textTheme.bodySmall?.copyWith(
                                              color: Colors.green,
                                              fontWeight: FontWeight.w500,
                                            ),
                                      ),
                                    ],
                                  ),
                                ),
                                IconButton(
                                  onPressed: () => setState(() {
                                    _selectedFileBytes = null;
                                    _selectedFileName = null;
                                  }),
                                  icon: Icon(
                                    Icons.close,
                                    color: Colors.white.withOpacity(0.7),
                                  ),
                                ),
                              ],
                            ),
                          ).animate().fadeIn().scale(),
                        ],
                      ],
                    ),
                  );
                },
              ).animate().fadeIn(delay: 200.ms).slideY(begin: 0.3),

              const SizedBox(height: 32),

              // Form Fields
              Container(
                padding: const EdgeInsets.all(32),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [
                      Colors.white.withOpacity(0.1),
                      Colors.white.withOpacity(0.05),
                    ],
                  ),
                  borderRadius: BorderRadius.circular(24),
                  border: Border.all(color: Colors.white.withOpacity(0.1)),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Beat Details',
                      style: Theme.of(context).textTheme.titleLarge?.copyWith(
                            color: Colors.white,
                            fontWeight: FontWeight.w600,
                          ),
                    ),
                    const SizedBox(height: 24),

                    // Title Field
                    _buildStyledTextField(
                      controller: _titleController,
                      label: 'Beat Title',
                      icon: Icons.music_note_outlined,
                      validator: (value) {
                        if (value == null || value.isEmpty) {
                          return 'Please enter a title';
                        }
                        return null;
                      },
                    ).animate().fadeIn(delay: 400.ms).slideX(begin: -0.2),

                    const SizedBox(height: 20),

                    // Description Field
                    _buildStyledTextField(
                      controller: _descriptionController,
                      label: 'Description (Optional)',
                      icon: Icons.description_outlined,
                      maxLines: 4,
                    ).animate().fadeIn(delay: 500.ms).slideX(begin: 0.2),
                  ],
                ),
              ).animate().fadeIn(delay: 300.ms).slideY(begin: 0.3),

              const SizedBox(height: 32),

              // Upload Button
              Container(
                width: double.infinity,
                height: 64,
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [
                      Theme.of(context).colorScheme.primary,
                      Theme.of(context).colorScheme.primary.withOpacity(0.8),
                    ],
                  ),
                  borderRadius: BorderRadius.circular(20),
                  boxShadow: [
                    BoxShadow(
                      color: Theme.of(context).colorScheme.primary.withOpacity(0.4),
                      blurRadius: 20,
                      spreadRadius: 3,
                      offset: const Offset(0, 6),
                    ),
                  ],
                ),
                child: ElevatedButton(
                  onPressed: _isUploading ? null : _uploadBeat,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.transparent,
                    shadowColor: Colors.transparent,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(20),
                    ),
                  ),
                  child: _isUploading
                      ? Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Container(
                              width: 32,
                              height: 32,
                              child: CircularProgressIndicator(
                                color: Colors.white,
                                strokeWidth: 3,
                                value: _uploadProgress,
                              ),
                            ),
                            const SizedBox(height: 8),
                            Text(
                              'Uploading... ${(_uploadProgress * 100).round()}%',
                              style: const TextStyle(
                                color: Colors.white,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ],
                        )
                      : Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            const Icon(Icons.rocket_launch, color: Colors.white, size: 28),
                            const SizedBox(width: 12),
                            Text(
                              'Upload Beat',
                              style: Theme.of(context).textTheme.titleLarge?.copyWith(
                                    color: Colors.white,
                                    fontWeight: FontWeight.w700,
                                    fontSize: 20,
                                  ),
                            ),
                          ],
                        ),
                ),
              ).animate().fadeIn(delay: 600.ms).slideY(begin: 0.3),

              const SizedBox(height: 24),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildStyledTextField({
    required TextEditingController controller,
    required String label,
    required IconData icon,
    int maxLines = 1,
    String? Function(String?)? validator,
  }) {
    return Container(
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [
            Colors.white.withOpacity(0.05),
            Colors.white.withOpacity(0.02),
          ],
        ),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white.withOpacity(0.1)),
      ),
      child: TextFormField(
        controller: controller,
        style: const TextStyle(color: Colors.white),
        maxLines: maxLines,
        decoration: InputDecoration(
          labelText: label,
          labelStyle: TextStyle(color: Colors.white.withOpacity(0.7)),
          prefixIcon: Icon(icon, color: Theme.of(context).colorScheme.primary),
          border: InputBorder.none,
          contentPadding: const EdgeInsets.all(20),
        ),
        validator: validator,
      ),
    );
  }
}

class WavePainter extends CustomPainter {
  final double progress;
  final Color color;

  WavePainter({required this.progress, required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.fill;

    final path = Path();
    final waveHeight = 20.0;
    final waveLength = size.width / 2;

    path.moveTo(0, size.height / 2);

    for (double x = 0; x <= size.width; x += 1) {
      final y = size.height / 2 + 
          waveHeight * math.sin((x / waveLength * 2 * math.pi) + (progress * 2 * math.pi));
      path.lineTo(x, y);
    }

    path.lineTo(size.width, size.height);
    path.lineTo(0, size.height);
    path.close();

    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
} 